// =============================================================================
// ToolMiddleware - 工具执行级中间件（装饰器模式）
// =============================================================================
use crate::ai::ports::tool::ToolExecutor;
use std::sync::Arc;

/// 工具执行级中间件：以装饰器方式组合 `ports::ToolExecutor`（鉴权/审计等）。
///
/// `wrap(inner)` 返回一个包装后的 executor；多层中间件通过嵌套 `wrap` 形成
/// 洋葱链，与 `RequestMiddleware` / `ports::llm::LoggingLlmClient` 的装饰器
/// 模式一致：
/// - 装饰后的 executor 是普通 `ToolExecutor`，可跨多次 execute 复用；
/// - 鉴权失败可直接短路返回，不调用 inner；
/// - 审计在调用 `inner.execute(app, calls)` 前后记录。
pub trait ToolMiddleware: Send + Sync {
    fn name(&self) -> &'static str;

    /// 将 `inner` 装饰为一个新的 `ToolExecutor`。实现方在 `execute` 中：
    /// - 前置校验（鉴权）：失败直接返回，不调用 inner；
    /// - 后置处理（审计/指标）：在调用 `inner.execute(app, calls)` 后记录。
    fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor>;
}

/// 将 `Vec<Arc<dyn ToolMiddleware>>` 折叠为单一 `ToolExecutor`。
///
/// - `middlewares[0]` 为最外层，与 `RequestMiddleware` 的 `build_llm_client_chain` 保持一致；
/// - 空 `middlewares` 时直接返回 `inner`（零开销默认路径）；
/// - 返回的 executor 可跨多次 `execute` 复用，语义与手动 `wrap` 嵌套等价：
///   `build_tool_executor_chain(vec![m1,m2], inner) == m1.wrap(m2.wrap(inner))`。
pub(crate) fn build_tool_executor_chain(
    middlewares: Vec<Arc<dyn ToolMiddleware>>,
    inner: Box<dyn ToolExecutor>,
) -> Box<dyn ToolExecutor> {
    middlewares
        .into_iter()
        .rev()
        .fold(inner, |acc, mw| mw.wrap(acc))
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::ToolMiddleware;
    use crate::ai::middleware::test_util::test_app;
    use crate::ai::ports::tool::{ToolExecOutput, ToolExecutor};
    use crate::ai::types::{App, FunctionCall, ToolCall};

    type BoxedExecErr = Box<dyn std::error::Error + Send + Sync>;
    type ExecFut<'a> = Pin<Box<dyn Future<Output = Result<ToolExecOutput, BoxedExecErr>> + Send + 'a>>;

    fn mock_calls() -> Vec<ToolCall> {
        vec![ToolCall {
            id: "call_1".into(),
            tool_type: "function".into(),
            function: FunctionCall {
                name: "read_file".into(),
                arguments: "{}".into(),
            },
        }]
    }

    /// 计数 executor：记录调用次数并返回空成功结果。
    struct CountingExecutor {
        calls: Arc<AtomicUsize>,
    }
    impl ToolExecutor for CountingExecutor {
        fn execute<'a>(&'a self, _app: &'a mut App, _calls: Vec<ToolCall>) -> ExecFut<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(ToolExecOutput::default()) })
        }
    }

    /// 计数中间件：每次 execute 递增计数并委托 inner（模拟审计层）。
    struct CountingMiddleware {
        calls: Arc<AtomicUsize>,
    }
    impl ToolMiddleware for CountingMiddleware {
        fn name(&self) -> &'static str { "counting" }
        fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
            let calls = Arc::clone(&self.calls);
            struct CountingExecutor {
                inner: Box<dyn ToolExecutor>,
                calls: Arc<AtomicUsize>,
            }
            impl ToolExecutor for CountingExecutor {
                fn execute<'a>(&'a self, app: &'a mut App, calls: Vec<ToolCall>) -> ExecFut<'a> {
                    let counter = Arc::clone(&self.calls);
                    Box::pin(async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        self.inner.execute(app, calls).await
                    })
                }
            }
            Box::new(CountingExecutor { inner, calls })
        }
    }

    /// 鉴权短路中间件：直接返回错误，不调用 inner。
    struct AuthMiddleware;
    impl ToolMiddleware for AuthMiddleware {
        fn name(&self) -> &'static str { "auth" }
        fn wrap(&self, _inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor> {
            struct AuthExecutor;
            impl ToolExecutor for AuthExecutor {
                fn execute<'a>(&'a self, _app: &'a mut App, _calls: Vec<ToolCall>) -> ExecFut<'a> {
                    Box::pin(async move { Err("auth denied".into()) })
                }
            }
            Box::new(AuthExecutor)
        }
    }

    /// 多层装饰链可组合，且装饰后的 executor 可跨多次 execute 复用。
    #[tokio::test]
    async fn decorator_chain_composes_and_reuses() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let outer_calls = Arc::new(AtomicUsize::new(0));

        // 外层计数 -> 内层计数 -> 真实 executor 的洋葱链
        let executor: Box<dyn ToolExecutor> = CountingMiddleware { calls: Arc::clone(&outer_calls) }
            .wrap(CountingMiddleware { calls: Arc::clone(&inner_calls) }
                .wrap(Box::new(CountingExecutor { calls: Arc::clone(&inner_calls) })));

        let mut app = test_app();
        for _ in 0..2 {
            let res = executor.execute(&mut app, mock_calls()).await;
            assert!(res.is_ok(), "计数 executor 固定成功");
        }
        assert_eq!(outer_calls.load(Ordering::SeqCst), 2);
        assert_eq!(inner_calls.load(Ordering::SeqCst), 4, "内层中间件与真实 executor 各被调用 2 次");
    }

    /// 鉴权短路中间件不调用 inner。
    #[tokio::test]
    async fn auth_middleware_skips_inner() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let executor: Box<dyn ToolExecutor> =
            AuthMiddleware.wrap(Box::new(CountingExecutor { calls: Arc::clone(&inner_calls) }));

        let mut app = test_app();
        let res = executor.execute(&mut app, mock_calls()).await;
        assert!(res.is_err());
        assert_eq!(inner_calls.load(Ordering::SeqCst), 0, "鉴权失败不应调用 inner");
    }
}