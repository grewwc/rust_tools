// =============================================================================
// RequestMiddleware - LLM request-level middleware (decorator pattern)
// =============================================================================
use crate::ai::ports::llm::LlmClient;
use std::sync::Arc;

/// LLM request-level middleware: composes `ports::LlmClient` in a decorator style
/// (retry, circuit-breaking, mock, etc.).
///
/// `wrap(inner)` returns a wrapped client; multiple middleware layers nest `wrap`
/// calls into an onion chain, e.g. `retry.wrap(logging.wrap(Box::new(DefaultLlmClient)))`
/// is equivalent to `Retry(Logging(Default))`, matching the generic decorator pattern
/// of `ports::llm::LoggingLlmClient`. Compared with a one-shot `next` closure design,
/// this signature guarantees:
/// - the decorated client is an ordinary `LlmClient` and can be reused across requests;
/// - retry/circuit-breaking middleware may call `inner.send` repeatedly (via `&self`),
///   and each call traverses the full inner decorator chain;
/// - mock middleware can short-circuit and return directly without calling inner.
pub trait RequestMiddleware: Send + Sync {
    fn name(&self) -> &'static str;

    /// Decorates `inner` into a new `LlmClient`. Implementations in `send`:
    /// - pre/post processing (logging, audit, metrics): hook around `inner.send(app, req)`;
    /// - short-circuit (mock, circuit breaker hit): return directly without calling inner;
    /// - retry: call `inner.send` repeatedly until success or the attempt limit.
    fn wrap(&self, inner: Box<dyn LlmClient>) -> Box<dyn LlmClient>;
}

/// Folds a `Vec<Arc<dyn RequestMiddleware>>` into a single `LlmClient` (onion model).
///
/// - `middlewares[0]` is the outermost layer (wrapped first), `middlewares.last()` sits
///   closest to inner;
/// - folding uses `rev().fold`, equivalent to
///   `middlewares[0].wrap(middlewares[1].wrap(...(inner)))`;
/// - the returned client is an ordinary `LlmClient` reusable across `send` calls, with
///   retry semantics fully preserved (see `decorator_chain_composes_and_reuses` /
///   `retry_*` in `tests`);
/// - an empty `middlewares` returns `inner` unchanged, keeping zero behavior change.
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

    fn mock_messages() -> Vec<Message> {
        vec![Message {
            role: "user".into(),
            content: Value::String("hi".into()),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }]
    }

    fn mock_req<'a>(messages: &'a [Message]) -> LlmRequest<'a> {
        LlmRequest {
            model: "mock-model".into(),
            messages,
            stream: false,
            tools_enabled: false,
        }
    }

    /// Counting client: records the call count and always returns an error
    /// (tests do not care about the real response).
    struct CountingClient {
        calls: Arc<AtomicUsize>,
    }
    impl LlmClient for CountingClient {
        fn send<'a>(&'a self, _app: &'a mut App, _req: LlmRequest<'a>) -> SendFut<'a> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async move { Err("mock client 固定失败".into()) })
        }
    }

    /// Counting middleware: increments the counter on each send and delegates to inner
    /// (mimics a logging/audit layer).
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
                fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest<'a>) -> SendFut<'a> {
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

    /// Retry middleware: calls inner repeatedly on failure (impossible with the old
    /// FnOnce `next` signature).
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
                fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest<'a>) -> SendFut<'a> {
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

    /// Short-circuit middleware: returns directly without calling inner.
    struct ShortCircuitMiddleware;
    impl RequestMiddleware for ShortCircuitMiddleware {
        fn name(&self) -> &'static str {
            "short-circuit"
        }
        fn wrap(&self, _inner: Box<dyn LlmClient>) -> Box<dyn LlmClient> {
            struct ShortCircuitClient;
            impl LlmClient for ShortCircuitClient {
                fn send<'a>(&'a self, _app: &'a mut App, _req: LlmRequest<'a>) -> SendFut<'a> {
                    Box::pin(async move { Err("short-circuit".into()) })
                }
            }
            Box::new(ShortCircuitClient)
        }
    }

    /// Multiple decorator layers compose, and the decorated client can be reused
    /// across requests.
    #[tokio::test]
    async fn decorator_chain_composes_and_reuses() {
        let outer_calls = Arc::new(AtomicUsize::new(0));
        let middleware_calls = Arc::new(AtomicUsize::new(0));
        let client_calls = Arc::new(AtomicUsize::new(0));

        // Logging(Retry(Default))-style onion chain: outer counter -> inner counter
        // -> real client.
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
        let msgs = mock_messages();
        for _ in 0..3 {
            let res = client.send(&mut app, mock_req(&msgs)).await;
            assert!(res.is_err(), "mock 固定失败，链应原样透传");
        }
        // The three layers count independently: outer middleware, inner middleware,
        // and the real client should each be called exactly 3 times.
        assert_eq!(outer_calls.load(Ordering::SeqCst), 3);
        assert_eq!(middleware_calls.load(Ordering::SeqCst), 3);
        assert_eq!(client_calls.load(Ordering::SeqCst), 3);
    }

    /// Retry middleware can call inner multiple times (the old FnOnce signature
    /// could not implement retry).
    #[tokio::test]
    async fn retry_middleware_calls_inner_multiple_times() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let client: Box<dyn LlmClient> =
            RetryMiddleware { max_attempts: 3 }.wrap(Box::new(CountingClient {
                calls: Arc::clone(&inner_calls),
            }));

        let mut app = test_app();
        let msgs = mock_messages();
        let res = client.send(&mut app, mock_req(&msgs)).await;
        assert!(res.is_err(), "固定失败，重试耗尽后冒泡错误");
        assert_eq!(
            inner_calls.load(Ordering::SeqCst),
            3,
            "重试中间件应多次调用 inner"
        );
    }

    /// Short-circuit middleware (mock/circuit breaker) does not call inner.
    #[tokio::test]
    async fn short_circuit_middleware_skips_inner() {
        let inner_calls = Arc::new(AtomicUsize::new(0));
        let client: Box<dyn LlmClient> = ShortCircuitMiddleware.wrap(Box::new(CountingClient {
            calls: Arc::clone(&inner_calls),
        }));

        let mut app = test_app();
        let msgs = mock_messages();
        let res = client.send(&mut app, mock_req(&msgs)).await;
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
                    fn send<'a>(&'a self, app: &'a mut App, req: LlmRequest<'a>) -> SendFut<'a> {
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
        let msgs = mock_messages();
        let _ = client.send(&mut app, mock_req(&msgs)).await;
        let got = order.lock().unwrap().clone();
        assert_eq!(
            got,
            vec!["enter:outer", "enter:inner", "exit:inner", "exit:outer"]
        );
        assert_eq!(inner_calls.load(Ordering::SeqCst), 1);
        order.lock().unwrap().clear();
        let _ = client.send(&mut app, mock_req(&msgs)).await;
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
        let msgs = mock_messages();
        let _ = client.send(&mut app, mock_req(&msgs)).await;
        assert_eq!(inner_calls.load(Ordering::SeqCst), 1);
    }
}
