// =============================================================================
// LlmClient - LLM request port
// =============================================================================
// Previously `request/transport.rs`'s `do_request_messages` was tightly coupled to
// driver/iteration, leaving no room to insert retry, circuit-breaking, logging, mock,
// and other middleware. This trait decouples it.
use std::future::Future;
use std::pin::Pin;

use crate::ai::history::Message;

/// Unified LLM request description (decoupled from `request/builder.rs` internals).
#[derive(Debug, Clone)]
pub(crate) struct LlmRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub stream: bool,
    pub tools_enabled: bool,
}

/// Unified LLM response handle: passes through the `reqwest::Response` plus the model
/// name actually used.
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

/// Default implementation: delegates to `request::do_request_messages` /
/// `do_request_messages_without_tools`
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

/// Decorator: logging middleware (example showing how to add cross-cutting behavior
/// without touching the driver)
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

/// Convenience constructor: returns the default `LlmClient` (`DefaultLlmClient`) so the
/// driver can plug in with one line.
///
/// Keeps zero behavior change: the default chain still uses `DefaultLlmClient`; when
/// middleware is needed, wrap via `middleware::request::build_llm_client_chain`.
pub(crate) fn default_llm_client() -> Box<dyn LlmClient> {
    Box::new(DefaultLlmClient)
}
