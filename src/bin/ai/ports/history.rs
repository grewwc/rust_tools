// =============================================================================
// HistoryStore - 历史存储端口（依赖倒置）
// =============================================================================
// 之前 driver 直接调用 `crate::ai::history::{blob, sqlite::*}` 的具体函数，
// 无法插入审计/加密/mock 等横切逻辑。现通过 trait 解耦，driver 只依赖抽象。
use std::{io, path::{Path, PathBuf}};

use crate::ai::history::Message;

// =============================================================================
// Compressor - 历史压缩策略端口（可插拔）
// =============================================================================
/// 可插拔压缩器：将已加载的 messages 按预算裁剪/摘要。
/// 默认实现委托现有 `history::compress` 逻辑，Noop 实现用于测试/旁路。
/// 该 trait 设计为对象安全，支持 `Box<dyn Compressor>` 注入。
pub(crate) trait Compressor: Send + Sync {
    fn compress(&self, messages: Vec<Message>, max_chars: usize, keep_last: usize) -> Vec<Message>;
    fn name(&self) -> &str;
}

/// 默认压缩器：委托现有 `history::compress_messages_for_context`（summary=0 简化版）。
/// 零行为变更默认路径仍走 `HistoryStore::build_context`，此实现仅在显式注入时生效；
/// 如需完整 summary/overflow 能力，可在调用侧构造携带额外参数的自定义 Compressor。
pub(crate) struct DefaultCompressor;
impl Compressor for DefaultCompressor {
    fn compress(&self, messages: Vec<Message>, max_chars: usize, keep_last: usize) -> Vec<Message> {
        if max_chars == 0 || messages.is_empty() {
            return messages;
        }
        // 透传到现有压缩逻辑（summary_max_chars=0, 无 overflow 归档），保持与硬编码路径语义一致；
        // 完整路径仍由 HistoryStore::build_context 负责，零行为变更。
        crate::ai::history::compress_messages_for_context(messages, max_chars, keep_last, 0, None, None)
    }
    fn name(&self) -> &str { "default" }
}

/// 空操作压缩器：原样返回，不做任何裁剪/摘要，用于测试或禁用压缩。
pub(crate) struct NoopCompressor;
impl Compressor for NoopCompressor {
    fn compress(&self, messages: Vec<Message>, _max_chars: usize, _keep_last: usize) -> Vec<Message> {
        messages
    }
    fn name(&self) -> &str { "noop" }
}

/// 历史存储端口：对象安全、最小化、不泄露 SQLite/文本双后端细节。
/// 保持 `pub(crate)` 以避免将内部 `RequestError` 等私有类型泄露到公共 API。
pub(crate) trait HistoryStore: Send + Sync {
    /// 读取可发送给模型的上下文投影（已做压缩、裁剪、溢出归档）。
    /// `cwd` 用于相对路径的 overflow archive 复用判定；不可用时传 `None`
    /// （与底层 `build_context_history` 的 `cwd: Option<&Path>` 契约一致）。
    fn build_context(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>>;

    /// 可插拔压缩变体：允许调用方注入 `Compressor` 策略。
    /// 默认实现保持向后兼容——忽略 compressor 透传到 `build_context`，保证零行为变更。
    /// 具体 `HistoryStore` 可重写此方法以真正应用 `compressor.compress`。
    fn build_context_with_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
        compressor: &dyn Compressor,
    ) -> io::Result<Vec<Message>> {
        let _ = compressor;
        self.build_context(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
        )
    }

    /// `Box<dyn Compressor>` 便捷重载，便于 `Pipeline` 中按所有权注入。
    fn build_context_with_boxed_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
        compressor: Box<dyn Compressor>,
    ) -> io::Result<Vec<Message>> {
        self.build_context_with_compressor(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
            compressor.as_ref(),
        )
    }

    /// 追加消息到 canonical 历史（原子写 + 溢出归档）。
    fn append_messages(&self, history_file: &Path, msgs: &[Message]) -> io::Result<()>;

    /// 模型感知的追加：sqlite 后端额外记录 source_model 溯源。
    /// 默认实现退化到 `append_messages`，对不关心模型溯源的自定义 store 零破坏。
    fn append_messages_for_model(
        &self,
        history_file: &Path,
        msgs: &[Message],
        source_model: &str,
    ) -> io::Result<()> {
        let _ = source_model;
        self.append_messages(history_file, msgs)
    }

    /// 加载原始历史（用于调试 / 重播）。
    fn load_messages(&self, history_file: &Path) -> io::Result<Vec<Message>>;
}

/// 默认实现：委托给现有 `history` 模块的具体函数，保持行为 100% 一致。
pub(crate) struct DefaultHistoryStore;

impl HistoryStore for DefaultHistoryStore {
    fn build_context(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        history_summary_max_chars: usize,
        overflow_dir: Option<PathBuf>,
        cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>> {
        crate::ai::history::build_context_history(
            history_count,
            history_file,
            history_max_chars,
            history_keep_last,
            history_summary_max_chars,
            overflow_dir,
            cwd,
        )
        .map_err(|e| {
            // 保留原始 io::ErrorKind（如 WouldBlock）以支持 snapshot 重试等上层逻辑
            if e.is::<io::Error>() {
                match e.downcast::<io::Error>() {
                    Ok(io_err) => *io_err,
                    Err(e2) => io::Error::new(io::ErrorKind::Other, e2.to_string()),
                }
            } else {
                io::Error::new(io::ErrorKind::Other, e.to_string())
            }
        })
    }

    fn build_context_with_compressor(
        &self,
        history_count: usize,
        history_file: &Path,
        history_max_chars: usize,
        history_keep_last: usize,
        _history_summary_max_chars: usize,
        _overflow_dir: Option<PathBuf>,
        _cwd: Option<&Path>,
        compressor: &dyn Compressor,
    ) -> io::Result<Vec<Message>> {
        // 演示“真正插拔”：先加载原始历史，再委派 compressor。
        // 与 `build_context` 的完整 sqlite+snapshop+cache 路径不同，此路径用于
        // pipeline/测试中显式注入策略；默认业务仍走 `build_context`，零行为变更。
        // 为避免与 cache 语义冲突，这里不走 `build_context_history` 的 cache，直接读取。
        let _ = history_count; // 可插拔路径不按 history_count 截断，压缩器自行按 keep_last 决策
        let messages = self.load_messages(history_file)?;
        Ok(compressor.compress(messages, history_max_chars, history_keep_last))
    }

    fn append_messages(&self, history_file: &Path, msgs: &[Message]) -> io::Result<()> {
        crate::ai::history::append_history_messages(history_file, msgs)
    }

    fn append_messages_for_model(
        &self,
        history_file: &Path,
        msgs: &[Message],
        source_model: &str,
    ) -> io::Result<()> {
        // sqlite 后端把 source_model 写入 meta 列（模型溯源）；blob 后端退化到普通追加。
        crate::ai::history::append_history_messages_for_model(history_file, msgs, source_model)
    }

    fn load_messages(&self, history_file: &Path) -> io::Result<Vec<Message>> {
        // 正确分发 sqlite / blob 双后端（通过 build_message_arr 内部的 is_sqlite_path 判定），
        // 避免将 sqlite 二进制用 read_to_string 静默丢弃为 empty。
        crate::ai::history::build_message_arr(usize::MAX, history_file).map_err(|e| {
            if e.is::<io::Error>() {
                match e.downcast::<io::Error>() {
                    Ok(io_err) => *io_err,
                    Err(e2) => io::Error::new(io::ErrorKind::Other, e2.to_string()),
                }
            } else {
                io::Error::new(io::ErrorKind::Other, e.to_string())
            }
        })
    }
}

/// 内存实现：用于测试 / 中间件单测，不触及文件系统。
#[cfg(test)]
pub(crate) struct InMemoryHistoryStore {
    pub(crate) messages: std::sync::Mutex<Vec<Message>>,
}

#[cfg(test)]
impl HistoryStore for InMemoryHistoryStore {
    fn build_context(
        &self,
        _history_count: usize,
        _history_file: &Path,
        _history_max_chars: usize,
        _history_keep_last: usize,
        _history_summary_max_chars: usize,
        _overflow_dir: Option<PathBuf>,
        _cwd: Option<&Path>,
    ) -> io::Result<Vec<Message>> {
        Ok(self.messages.lock().unwrap().clone())
    }
    fn append_messages(&self, _history_file: &Path, msgs: &[Message]) -> io::Result<()> {
        self.messages.lock().unwrap().extend_from_slice(msgs);
        Ok(())
    }
    fn load_messages(&self, _history_file: &Path) -> io::Result<Vec<Message>> {
        Ok(self.messages.lock().unwrap().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::history::Message;

    #[test]
    fn in_memory_store_append_messages_for_model_falls_back_to_plain_append() {
        let store = InMemoryHistoryStore {
            messages: std::sync::Mutex::new(Vec::new()),
        };
        let msgs = vec![Message {
            role: "user".into(),
            content: serde_json::json!("hi"),
            tool_calls: None,
            tool_call_id: None,
            reasoning_content: None,
        }];
        // Step 6：trait 默认实现忽略 source_model、退化到 `append_messages`，
        // 对不关心模型溯源的自定义 store 零破坏。
        store
            .append_messages_for_model(Path::new("unused"), &msgs, "some-model")
            .unwrap();
        assert_eq!(store.messages.lock().unwrap().len(), 1);
    }
}
