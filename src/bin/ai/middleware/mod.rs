// =============================================================================
// Middleware - 中间件抽象（全局解耦核心）
// =============================================================================
// 统一的中间件模型：任意横切逻辑（日志、审计、限流、重试、敏感词过滤、压缩
// 策略等）通过装饰器组合到端口上，无需改动 driver / request / history 的硬编码路径。
//
// 两层模型（有意分离，不合并）：
// - RequestMiddleware (request.rs)       : 装饰 `ports::LlmClient::send`，用于重试/熔断/mock LLM 调用
// - ToolMiddleware    (tool.rs)          : 装饰 `ports::ToolExecutor::execute`，用于鉴权/审计工具调用
// 通过端口装饰器（类似 LoggingLlmClient）组合，保持关注点分离：LLM/工具层可独立测试与替换。
//
// 灵感：tower::Layer + axum::middleware，但保持零外部依赖、全同步/异步兼容。

pub(crate) mod request;
pub(crate) mod tool;

#[cfg(test)]
pub(crate) mod test_util;

#[allow(unused_imports)]
pub(crate) use request::RequestMiddleware;
#[allow(unused_imports)]
pub(crate) use tool::ToolMiddleware;

