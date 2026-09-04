// =============================================================================
// ToolMiddleware - tool execution-level middleware (decorator pattern)
// =============================================================================
use crate::ai::ports::tool::ToolExecutor;
use std::sync::Arc;

/// Tool execution-level middleware: composes `ports::ToolExecutor` via decorators (auth/audit, etc.).
///
/// `wrap(inner)` returns a wrapped executor; multiple middleware layers form an
/// onion chain via nested `wrap` calls, matching the decorator pattern used by
/// `RequestMiddleware` / `ports::llm::LoggingLlmClient`:
/// - The decorated executor is a regular `ToolExecutor` reusable across multiple execute calls;
/// - Auth failures short-circuit and return without calling inner;
/// - Auditing happens before and after calling `inner.execute(app, calls)`.
pub trait ToolMiddleware: Send + Sync {
    fn name(&self) -> &'static str;

    /// Decorates `inner` into a new `ToolExecutor`. Implementations, inside `execute`:
    /// - Pre-checks (auth): return immediately on failure without calling inner;
    /// - Post-processing (audit/metrics): record after calling `inner.execute(app, calls)`.
    fn wrap(&self, inner: Box<dyn ToolExecutor>) -> Box<dyn ToolExecutor>;
}

/// Folds a `Vec<Arc<dyn ToolMiddleware>>` into a single `ToolExecutor`.
///
/// - `middlewares[0]` is the outermost, consistent with `RequestMiddleware`'s `build_llm_client_chain`;
/// - An empty `middlewares` returns `inner` directly (zero-cost default path);
/// - The returned executor is reusable across multiple `execute` calls, semantically
///   equivalent to manual `wrap` nesting: `build_tool_executor_chain(vec![m1,m2], inner) == m1.wrap(m2.wrap(inner))`.
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
    type ExecFut<'a> =
        Pin<Box<dyn Future<Output = Result<ToolExecOutput, BoxedExecErr>> + Send + 'a>>;

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

    /// Counting executor: records the number of calls and returns an empty successful result.
    struct CountingExecutor {
        calls: Arc<AtomicUsize>,
    }
    impl ToolExecutor for CountingExecutor {
        fn execute<'a>(&'a self, _app: &'a mut App, _calls: Vec<ToolCall>) -> ExecFut<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Ok(ToolExecOutput::default()) })
        }
    }

    /// Counting middleware: increments a counter on each execute and delegates to inner (simulating an audit layer).
    struct CountingMiddleware {
        calls: Arc<AtomicUsize>,
    }
    impl ToolMiddleware for CountingMiddleware {
        fn name(&self) -> &'static str {
            "counting"
        }
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

    /// Auth short-circuit middleware: returns an error directly without calling inner.
    struct AuthMiddleware;
    impl ToolMiddleware for AuthMiddleware {
        fn name(&self) -> &'static str {
            "auth"
        }
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

    /// A multi-layer decorator chain composes, and the decorated executor is reusable across multiple execute calls.
    #[tokio::test]
    async fn decorator_chain_composes_and_reuses() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let outer_calls = Arc::new(AtomicUsize::new(0));

        // Onion chain: outer counter -> inner counter -> real executor
        let executor: Box<dyn ToolExecutor> = CountingMiddleware {
            calls: Arc::clone(&outer_calls),
        }
        .wrap(
            CountingMiddleware {
                calls: Arc::clone(&inner_calls),
            }
            .wrap(Box::new(CountingExecutor {
                calls: Arc::clone(&inner_calls),
            })),
        );

        let mut app = test_app();
        for _ in 0..2 {
            let res = executor.execute(&mut app, mock_calls()).await;
            assert!(res.is_ok(), "计数 executor 固定成功");
        }
        assert_eq!(outer_calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            inner_calls.load(Ordering::SeqCst),
            4,
            "内层中间件与真实 executor 各被调用 2 次"
        );
    }

    /// Auth short-circuit middleware does not call inner.
    #[tokio::test]
    async fn auth_middleware_skips_inner() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let executor: Box<dyn ToolExecutor> = AuthMiddleware.wrap(Box::new(CountingExecutor {
            calls: Arc::clone(&inner_calls),
        }));

        let mut app = test_app();
        let res = executor.execute(&mut app, mock_calls()).await;
        assert!(res.is_err());
        assert_eq!(
            inner_calls.load(Ordering::SeqCst),
            0,
            "鉴权失败不应调用 inner"
        );
    }
}
