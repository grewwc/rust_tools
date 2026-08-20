// =============================================================================
// LlmClient - LLM 请求端口
// =============================================================================
// 之前 `request/transport.rs` 的 `do_request_messages` 直接被 driver/iteration 强耦合，
// 无法插入重试、熔断、日志、mock 等中间件。现通过此 trait 解耦。
use std::future::Future;
use std::pin::Pin;

use crate::ai::history::Message;

/// 统一的 LLM 请求描述（与 `request/builder.rs` 的内部结构解耦）。
#[derive(Debug, Clone)]
pub(crate) struct LlmRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub tools_enabled: bool,
}

/// 统一的 LLM 响应句柄：透传 `reqwest::Response` + 实际使用的 model 名。
pub(crate) struct LlmResponse {
    pub response: reqwest::Response,
    pub model: String,
}

pub(crate) trait LlmClient: Send + Sync {
    fn send<'a>(
        &'a self,
        app: &'a mut crate::ai::types::App,
        req: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>>;
}

/// 默认实现：委托给 `request::do_request_messages` / `do_request_messages_without_tools`
pub(crate) struct DefaultLlmClient;

impl LlmClient for DefaultLlmClient {
    fn send<'a>(
        &'a self,
        app: &'a mut crate::ai::types::App,
        req: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        Box::pin(async move {
            let model = req.model.clone();
            let raw = if req.tools_enabled {
                crate::ai::request::do_request_messages(app, &model, &req.messages, req.stream).await
            } else {
                crate::ai::request::do_request_messages_without_tools(
                    app, &model, &req.messages, req.stream,
                )
                .await
            }
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;
            Ok(LlmResponse { response: raw, model })
        })
    }
}

/// 装饰器：日志中间件（示例，展示如何无需改 driver 即可插横切）
pub(crate) struct LoggingLlmClient<C: LlmClient> {
    inner: C,
}

impl<C: LlmClient> LoggingLlmClient<C> {
    pub fn new(inner: C) -> Self { Self { inner } }
}

impl<C: LlmClient> LlmClient for LoggingLlmClient<C> {
    fn send<'a>(
        &'a self,
        app: &'a mut crate::ai::types::App,
        req: LlmRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, Box<dyn std::error::Error + Send + Sync>>> + Send + 'a>> {
        let req_clone = req.clone();
        let fut = self.inner.send(app, req);
        Box::pin(async move {
            eprintln!("[llm] -> {} ({} msgs, stream={})", req_clone.model, req_clone.messages.len(), req_clone.stream);
            let res = fut.await;
            match &res {
                Ok(_) => eprintln!("[llm] <- {} ok", req_clone.model),
                Err(e) => eprintln!("[llm] <- {} err: {e}", req_clone.model),
            }
            res
        })
    }
}

/// 便捷构造：返回默认 `LlmClient`（`DefaultLlmClient`），供 driver 一行接入。
///
/// 保持零行为变更：默认链仍走 `DefaultLlmClient` 行为；需要中间件时
/// 通过 `middleware::request::build_llm_client_chain` 包装。
pub(crate) fn default_llm_client() -> Box<dyn LlmClient> {
    Box::new(DefaultLlmClient)
}
