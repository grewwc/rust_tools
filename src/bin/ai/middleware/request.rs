// =============================================================================
// RequestMiddleware - LLM 请求级中间件（装饰器模式）
// =============================================================================
use crate::ai::ports::llm::LlmClient;
use std::sync::Arc;

/// LLM 请求级中间件：以装饰器方式组合 `ports::LlmClient`（重试/熔断/mock 等）。
///
/// `wrap(inner)` 返回一个包装后的 client；多层中间件通过嵌套 `wrap` 形成洋葱链，
/// 例如 `retry.wrap(logging.wrap(Box::new(DefaultLlmClient)))` 等价于
/// `Retry(Logging(Default))`，与 `ports::llm::LoggingLlmClient` 的泛型装饰器
/// 模式一致。相比"一次性 next 闭包"的设计，本签名保证：
/// - 装饰后的 client 是普通 `LlmClient`，可跨多次请求复用；
/// - 重试/熔断中间件可对 `inner.send` 反复调用（经 `&self`），且每次调用都
///   穿过内层完整装饰链；
/// - mock 中间件可短路直接返回，不调用 inner。
pub trait RequestMiddleware: Send + Sync {
    fn name(&self) -> &'static str;

    /// 将 `inner` 装饰为一个新的 `LlmClient`。实现方在 `send` 中：
    /// - 前置/后置处理（日志、审计、指标）：在调用 `inner.send(app, req)` 前后插入；
    /// - 短路（mock、熔断命中）：直接返回，不调用 inner；
    /// - 重试：可对 `inner.send` 多次调用直到成功或达到上限。
    fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient>;
}

/// 将 `Vec<Arc<dyn RequestMiddleware>>` 折叠为单一 `LlmClient`（洋葱模型）。
///
/// - `middlewares[0]` 为最外层（最先 wrap），`middlewares.last()` 最靠近 inner；
/// - 折叠顺序为 `rev().fold`，等价于 `middlewares[0].wrap(middlewares[1].wrap(...(inner)))`；
/// - 返回的 client 为普通 `LlmClient`，可跨多次 `send` 复用，重试语义完整保留
///  （见 `tests` 中 `decorator_chain_composes_and_reuses` / `retry_*`）。
/// - 空 `middlewares` 时直接返回 `inner`，保持零行为变更。
pub(crate) fn build_llm_client_chain(
    middlewares: Vec<Arc<dyn RequestMiddleware>>,
    inner: Box<dyn LlmClient>,
) -> Box<dyn LlmClient> {
    middlewares
        .into_iter()
        .rev()
        .fold(inner, |client, mw| mw.wrap(client))
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::RequestMiddleware;
    use crate::ai::history::Message;
    use crate::ai::middleware::test_util::test_app;
    use crate::ai::ports::llm::{LlmClient, LlmRequest, LlmResponse};
    use crate::ai::types::App;
    use serde_json::Value;

    type BoxedSendErr = Box<dyn std::error::Error + Send + Sync>;
    type SendFut<'a> = Pin<Box<dyn Future<Output = Result<LlmResponse, BoxedSendErr>> + Send + 'a>>;

    fn mock_req() -> LlmRequest {
        LlmRequest {
            model: "mock-model".into(),
            messages: vec![Message {
                role: "user".into(),
                content: Value::String("hi".into()),
                tool_calls: None,
                tool_call_id: None,
                reasoning_content: None,
            }],
            stream: false,
            tools_enabled: false,
        }
    }

    /// 计数 client：记录被调用次数并固定返回错误（测试不关心真实响应）。
    struct CountingClient {
        calls: Arc<AtomicUsize>,
    }
    impl LlmClient for CountingClient {
        fn send<'a>(&'a self, _app: &'a mut App, _req: LlmRequest) -> SendFut<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Err("mock client 固定失败".into()) })
        }
    }

    /// 计数中间件：每次 send 递增计数并委托 inner（模拟日志/审计层）。
    struct CountingMiddleware {
        calls: Arc<AtomicUsize>,
    }
    impl RequestMiddleware for CountingMiddleware {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            let calls = Arc::clone(&self.calls);
            struct CountingClient {
                inner: Box<dyn LlmClient>,
                calls: Arc<AtomicUsize>,
            }
            impl LlmClient for CountingClient {
                fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest) -> SendFut<'a> {
                    let calls = Arc::clone(&self.calls);
                    Box::pin(async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        self.inner.send(app, req).await
                    })
                }
            }
            Box::new(CountingClient { inner, calls })
        }
    }

    /// 重试中间件：失败时多次调用 inner（旧签名下 FnOnce next 无法实现）。
    struct RetryMiddleware {
        max_attempts: usize,
    }
    impl RequestMiddleware for RetryMiddleware {
        fn name(&self) -> &'static str {
            "retry"
        }
        fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            let max_attempts = self.max_attempts;
            struct RetryClient {
                inner: Box<dyn LlmClient>,
                max_attempts: usize,
            }
            impl LlmClient for RetryClient {
                fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest) -> SendFut<'a> {
                    let max_attempts = self.max_attempts;
                    Box::pin(async move {
                        let mut attempt = 0;
                        loop {
                            attempt += 1;
                            let res = self.inner.send(app, req.clone()).await;
                            if res.is_ok() || attempt >= max_attempts {
                                return res;
                            }
                        }
                    })
                }
            }
            Box::new(RetryClient {
                inner,
                max_attempts,
            })
        }
    }

    /// 短路中间件：直接返回，不调用 inner。
    struct ShortCircuitMiddleware;
    impl RequestMiddleware for ShortCircuitMiddleware {
        fn name(&self) -> &'static str {
            "short-circuit"
        }
        fn wrap(&self, _inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            struct ShortCircuitClient;
            impl LlmClient for ShortCircuitClient {
                fn send<'a>(&'a self, _app: &'a mut App, _req: LlmRequest) -> SendFut<'a> {
                    Box::pin(async move { Err("short-circuit".into()) })
                }
            }
            Box::new(ShortCircuitClient)
        }
    }

    /// 多层装饰链可组合，且装饰后的 client 可跨多次请求复用。
    #[tokio::test]
    async fn decorator_chain_composes_and_reuses() {
        let outer_calls = Arc::new(AtomicUsize::new(0));
        let middleware_calls = Arc::new(AtomicUsize::new(0));
        let client_calls = Arc::new(AtomicUsize::new(0));

        // Logging(Retry(Default)) 式洋葱链：外层计数 -> 内层计数 -> 真实 client
        let client: Box<dyn LlmClient> = CountingMiddleware {
            calls: Arc::clone(&outer_calls),
        }
        .wrap(
            CountingMiddleware {
                calls: Arc::clone(&middleware_calls),
            }
            .wrap(Box::new(CountingClient {
                calls: Arc::clone(&client_calls),
            })),
        );

        let mut app = test_app();
        for _ in 0..3 {
            let res = client.send(&mut app, mock_req()).await;
            assert!(res.is_err(), "mock 固定失败，链应原样透传");
        }
        // 三层各自独立计数：外层中间件、内层中间件、真实 client 各应恰被调用 3 次。
        assert_eq!(outer_calls.load(Ordering::SeqCst), 3);
        assert_eq!(middleware_calls.load(Ordering::SeqCst), 3);
        assert_eq!(client_calls.load(Ordering::SeqCst), 3);
    }

    /// 重试中间件可对 inner 多次调用（旧 FnOnce 签名无法实现重试）。
    #[tokio::test]
    async fn retry_middleware_calls_inner_multiple_times() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let client: Box<dyn LlmClient> =
            RetryMiddleware { max_attempts: 3 }.wrap(Box::new(CountingClient {
                calls: Arc::clone(&inner_calls),
            }));

        let mut app = test_app();
        let res = client.send(&mut app, mock_req()).await;
        assert!(res.is_err(), "固定失败，重试耗尽后冒泡错误");
        assert_eq!(
            inner_calls.load(Ordering::SeqCst),
            3,
            "重试中间件应多次调用 inner"
        );
    }

    /// 短路中间件（mock/熔断）不调用 inner。
    #[tokio::test]
    async fn short_circuit_middleware_skips_inner() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let client: Box<dyn LlmClient> = ShortCircuitMiddleware.wrap(Box::new(CountingClient {
            calls: Arc::clone(&inner_calls),
        }));

        let mut app = test_app();
        let res = client.send(&mut app, mock_req()).await;
        assert!(res.is_err());
        assert_eq!(
            inner_calls.load(Ordering::SeqCst),
            0,
            "短路中间件不应调用 inner"
        );
    }

    #[tokio::test]
    async fn build_chain_folds_onion_outer_first() {
        use super::build_llm_client_chain;
        use std::sync::Mutex;

        let order = Arc::new(Mutex::new(Vec::<String>::new()));
        struct OrderMiddleware {
            name: &'static str,
            order: Arc<Mutex<Vec<String>>>,
        }
        impl RequestMiddleware for OrderMiddleware {
            fn name(&self) -> &'static str {
                self.name
            }
            fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
                let name = self.name;
                let order = Arc::clone(&self.order);
                struct OrderClient {
                    inner: Box<dyn LlmClient>,
                    name: &'static str,
                    order: Arc<Mutex<Vec<String>>>,
                }
                impl LlmClient for OrderClient {
                    fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest) -> SendFut<'a> {
                        let name = self.name;
                        let order = Arc::clone(&self.order);
                        Box::pin(async move {
                            order.lock().unwrap().push(format!("enter:{name}"));
                            let res = self.inner.send(app, req).await;
                            order.lock().unwrap().push(format!("exit:{name}"));
                            res
                        })
                    }
                }
                Box::new(OrderClient { inner, name, order })
            }
        }

        let inner_calls = Arc::new(AtomicUsize::new(0));
        let middlewares: Vec<Arc<dyn RequestMiddleware>> = vec![
            Arc::new(OrderMiddleware {
                name: "outer",
                order: Arc::clone(&order),
            }),
            Arc::new(OrderMiddleware {
                name: "inner",
                order: Arc::clone(&order),
            }),
        ];
        let client = build_llm_client_chain(
            middlewares,
            Box::new(CountingClient {
                calls: Arc::clone(&inner_calls),
            }),
        );
        let mut app = test_app();
        let _ = client.send(&mut app, mock_req()).await;
        let got = order.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["enter:outer", "enter:inner", "exit:inner", "exit:outer"]
        );
        assert_eq!(inner_calls.load(Ordering::SeqCst), 1);
        order.lock().unwrap().clear();
        let _ = client.send(&mut app, mock_req()).await;
        assert_eq!(order.lock().unwrap().len(), 4);
    }

    #[tokio::test]
    async fn build_chain_empty_returns_inner() {
        use super::build_llm_client_chain;
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let client = build_llm_client_chain(
            vec![],
            Box::new(CountingClient {
                calls: Arc::clone(&inner_calls),
            }),
        );
        let mut app = test_app();
        let _ = client.send(&mut app, mock_req()).await;
        assert_eq!(inner_calls.load(Ordering::SeqCst), 1);
    }
}
