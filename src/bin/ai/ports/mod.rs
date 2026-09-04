// =============================================================================
// Ports - 核心域端口（Ports & Adapters / Hexagonal）
// =============================================================================
// 定义 AI runtime 的所有对外依赖抽象（History、LLM、Stream、Tool、Prompt），
// 让 driver / pipeline 只依赖 trait，不依赖具体实现。任意实现均可通过
// 中间件链替换、装饰或 mock，便于扩展与测试。
//
// 设计原则：
// - trait 位于 core，impl 位于各子系统（history、request、stream、tools...）
// - 保持最小化、对象安全、避免泄露具体存储/协议细节
// - 为中间件提供统一的挂载点

pub(crate) mod history;
pub(crate) mod llm;
pub(crate) mod mcp;
pub(crate) mod prompt;
pub(crate) mod stream;
pub(crate) mod tool;

#[allow(unused_imports)]
pub(crate) use history::{DefaultHistoryStore, HistoryStore};
#[allow(unused_imports)]
pub(crate) use llm::{DefaultLlmClient, LlmClient, LlmRequest, LlmResponse};
#[allow(unused_imports)]
pub(crate) use mcp::{InMemoryMcpPort, LiveMcpPort, McpPort, McpToolDef};
#[allow(unused_imports)]
pub(crate) use prompt::{DefaultPromptBuilder, PromptBuilder};
#[allow(unused_imports)]
pub(crate) use stream::{DecodedStream, StreamDecoder};
#[allow(unused_imports)]
pub(crate) use tool::{ToolExecOutput, ToolExecutor};
