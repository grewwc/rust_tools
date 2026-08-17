//! 流式线协议解析原语（provider 适配层与 stream 归一化层共享）。
//!
//! `ParsedStreamPayload` 原定义于 `stream/state.rs`，`try_parse_stream_chunk` /
//! `try_parse_stream_chunk_loose` 原定义于 `stream/normalize.rs`。provider 适配层
//! 需要这两个解析原语，而 stream 归一化层又依赖 provider 的 `ProviderAdapter`
//! trait，形成 provider ↔ stream 互环。解析原语下沉到此中立的 request 子模块后，
//! 双方统一经 `crate::ai::request` 引用，不再互相引用模块树。

use super::StreamChunk;

pub(in crate::ai) enum ParsedStreamPayload {
    Ignore,
    Done,
    Chunk(StreamChunk),
    /// content_part.added（output_text 类型）携带的是该 part 当前已存在的完整文本，
    /// 与 output_text.delta 增量重叠，属于协议多路径重发而非模型新增内容。
    /// 按增量格式解析（think_demux 拆分等仍生效），但流层会对 content 额外做
    /// 未见后缀去重，避免正文跨事件路径重复渲染。
    ReplayedChunk(StreamChunk),
    SnapshotChunk(StreamChunk),
    /// Responses 协议返回的完整 `reasoning` output item（含 `id` /
    /// `encrypted_content` / `summary`）。用于同 turn 工具链回放：原样透传给
    /// 后续请求的 input，使模型保留上一跳推理上下文。不进持久化历史。
    ReasoningItem(serde_json::Value),
    /// provider 在流中途返回了 error 对象或 error 事件，携带可读错误信息。
    Error(String),
}

fn try_parse_stream_chunk(payload: &str) -> Option<StreamChunk> {
    let mut chunk = serde_json::from_str::<StreamChunk>(payload).ok()?;
    chunk.merge_reasoning();
    Some(chunk)
}

pub(in crate::ai) fn try_parse_stream_chunk_loose(payload: &str) -> Option<StreamChunk> {
    if let Some(chunk) = try_parse_stream_chunk(payload) {
        return Some(chunk);
    }

    let trimmed = payload.trim();
    let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) else {
        return None;
    };
    if start >= end {
        return None;
    }

    let candidate = &trimmed[start..=end];
    try_parse_stream_chunk(candidate)
}
