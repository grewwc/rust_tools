//! 把内联在 `content` 通道里的推理链拆分回 reasoning 通道。
//!
//! 某些兼容端点（如火山方舟 `/coding`）的 chat 模板会在 assistant 预填位注入
//! `<think>`，使模型把推理链写进 `content` 通道、仅以悬空 `</think>` 收尾。
//! [`ContentThinkDemuxer`] 负责在 `content` 流里检测首个 `</think>`，把它之前
//! 的文本归入 reasoning 通道、之后的文本留在 content 通道。
//!
//! ## 非破坏性 withhold 设计
//!
//! 捕获态下 **不向任何通道增量吐出**：所有 content 暂存在缓冲区里，直到
//! `</think>` 到达才一次性把前缀作为 reasoning 提交。若 `</think>` 始终未到达
//! （arm 判定错误、fallback 到非预填模型、流中断），[`Self::flush`] 把整段缓冲按
//! **content** 安全回退——宁可降级为「思考泄漏进正文」也绝不丢失可见答案。
//!
//! 代价是捕获期间无增量 thinking 渲染；`</think>` 到达后推理链一次性出现在
//! thinking 折叠区，随后正文正常流式。对于预填模型这一延迟仅覆盖推理阶段。

const CLOSE_TAG: &str = "</think>";
const CLOSE_TAG_LEN: usize = CLOSE_TAG.len();
/// 捕获态缓冲区上限（字节）。推理链极少超过此量级；超限说明 `</think>` 大概率
/// 不会到达（arm 判定错误或 fallback 到非预填模型），此时放弃捕获、把已缓冲
/// 内容按正文冲刷——宁可降级为「思考泄漏到正文」也不丢失可见答案。
const CAPTURE_LIMIT: usize = 1 << 21; // 2 MiB

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// 捕获态：缓冲 content，等待首个 `</think>`。
    Capture,
    /// 直通态：content 原样放行（初始态 & `</think>` 闭合后）。
    Passthrough,
}

pub(super) struct ContentThinkDemuxer {
    mode: Mode,
    /// 捕获态下的完整缓冲区。未确认 `</think>` 前不向任何通道吐出（withhold），
    /// 从而在 `</think>` 始终未到达时可以把整段安全回退为 content。
    buffer: String,
}

impl ContentThinkDemuxer {
    pub(super) fn new() -> Self {
        Self {
            mode: Mode::Passthrough,
            buffer: String::new(),
        }
    }

    /// 置为捕获态：content 开头即被缓冲，直到首个 `</think>`。
    pub(super) fn arm(&mut self) {
        self.mode = Mode::Capture;
    }

    /// 喂入一个 content chunk，返回 `(reasoning, content)` 两路文本。
    /// 直通态是零拷贝语义的快速路径。
    pub(super) fn push(&mut self, chunk: &str) -> (String, String) {
        if self.mode == Mode::Passthrough {
            return (String::new(), chunk.to_string());
        }

        // Capture: withhold content into buffer, search for </think>.
        self.buffer.push_str(chunk);

        // 仅搜索新 chunk 与旧缓冲尾部的重叠区域（跨 chunk 标签）。
        // old buffer 已搜过且未命中，</think> 只可能始于旧缓冲最后 CLOSE_TAG_LEN-1
        // 字节并跨入新 chunk。
        let overlap = CLOSE_TAG_LEN - 1;
        let old_len = self.buffer.len() - chunk.len(); // push 前的合法字节边界
        let search_start = old_len.saturating_sub(overlap);
        // 字节索引可能落在多字节 UTF-8 字符（如中文推理链）内部，修正到字符边界，
        // 避免 `&buffer[search_start..]` 越界 panic。修正只会让搜索区域略变大，
        // 不会漏检：</think> 是纯 ASCII，其起点必然落在字符边界上。
        let search_start = self.buffer.floor_char_boundary(search_start);
        let region = &self.buffer[search_start..];
        if let Some(rel) = region.find(CLOSE_TAG) {
            let idx = search_start + rel;
            // 安全：idx + CLOSE_TAG_LEN <= buffer.len()（find 保证完整匹配）。
            let reasoning = self.buffer[..idx].to_string();
            let content = self.buffer[idx + CLOSE_TAG_LEN..].to_string();
            self.buffer.clear();
            self.mode = Mode::Passthrough;
            (reasoning, content)
        } else {
            // 缓冲区超限且本次仍未闭合：大概率不会再出现 </think>，放弃捕获，
            // 按正文冲刷。必须在搜索之后判断，避免同一大 chunk 尾部已有闭合标签时
            // 误把整段 reasoning 泄漏到可见正文。
            if self.buffer.len() > CAPTURE_LIMIT {
                return self.abort();
            }
            // 未命中：继续缓冲，不向任何通道吐出（withhold）。
            (String::new(), String::new())
        }
    }

    /// 放弃捕获：把已缓冲内容全部按正文冲刷，切回直通态。
    pub(super) fn abort(&mut self) -> (String, String) {
        let content = std::mem::take(&mut self.buffer);
        self.mode = Mode::Passthrough;
        (String::new(), content)
    }

    /// 流结束冲刷：若仍在捕获态（`</think>` 从未到达），把缓冲按正文回退。
    pub(super) fn flush(&mut self) -> (String, String) {
        if self.buffer.is_empty() {
            return (String::new(), String::new());
        }
        // 仍在捕获态 = </think> 未到达：安全回退为 content。
        self.abort()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_not_armed_is_noop() {
        let mut d = ContentThinkDemuxer::new();
        // 未 arm：即便 content 里含 </think> 也原样直通，其它模型行为不变。
        assert_eq!(
            d.push("answer </think> tail"),
            (String::new(), "answer </think> tail".to_string())
        );
        assert_eq!(d.flush(), (String::new(), String::new()));
    }

    #[test]
    fn splits_dangling_close_tag_in_single_chunk() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // 预填模板：无 <think> 开标签，仅悬空 </think> 收尾。
        assert_eq!(
            d.push("let me think</think>## Final answer"),
            ("let me think".to_string(), "## Final answer".to_string())
        );
        // 闭合后转直通：后续 chunk 全部作为可见正文。
        assert_eq!(d.push(" more"), (String::new(), " more".to_string()));
    }

    #[test]
    fn withholds_until_close_tag_then_commits() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // 捕获态：不增量吐出，整段缓冲（withhold）。
        assert_eq!(d.push("reasoning "), (String::new(), String::new()));
        assert_eq!(d.push("part</thi"), (String::new(), String::new()));
        // </think> 跨 chunk 补齐：一次性提交全部推理 + 正文。
        assert_eq!(
            d.push("nk>done"),
            ("reasoning part".to_string(), "done".to_string())
        );
    }

    #[test]
    fn flush_returns_buffered_as_content_when_no_close_tag() {
        // 核心安全测试：arm 判定错误或流中断时，</think> 从未到达，
        // 整段缓冲必须按 content 回退，绝不丢失进 reasoning。
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        assert_eq!(d.push("think</thi"), (String::new(), String::new()));
        assert_eq!(d.push(" more content"), (String::new(), String::new()));
        assert_eq!(
            d.flush(),
            (String::new(), "think</thi more content".to_string())
        );
    }

    #[test]
    fn flush_empty_after_close_tag() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        d.push("reasoning</think>answer");
        // 闭合后 flush 无残留。
        assert_eq!(d.flush(), (String::new(), String::new()));
    }

    #[test]
    fn close_tag_at_start_yields_empty_reasoning() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // </think> 在最前面：推理为空，全部是正文。
        assert_eq!(
            d.push("</think>immediate answer"),
            (String::new(), "immediate answer".to_string())
        );
    }

    #[test]
    fn capture_limit_aborts_as_content() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // 喂入超过 CAPTURE_LIMIT 的内容（不含 </think>）：超限后放弃捕获，按正文冲刷。
        let big = "x".repeat(CAPTURE_LIMIT + 1);
        let (reasoning, content) = d.push(&big);
        assert!(reasoning.is_empty());
        assert_eq!(content.len(), CAPTURE_LIMIT + 1);
        // 后续 chunk 直通。
        assert_eq!(d.push("tail"), (String::new(), "tail".to_string()));
    }

    #[test]
    fn close_tag_in_oversized_chunk_wins_over_capture_limit() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        let reasoning = "x".repeat(CAPTURE_LIMIT + 1);
        let input = format!("{reasoning}</think>answer");

        assert_eq!(d.push(&input), (reasoning, "answer".to_string()));
    }

    #[test]
    fn only_first_close_tag_splits() {
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // 首个 </think> 拆分后转直通，第二个 </think> 作为普通正文。
        assert_eq!(
            d.push("think</think>text</think>more"),
            ("think".to_string(), "text</think>more".to_string())
        );
    }

    #[test]
    fn chinese_reasoning_byte_offset_does_not_panic() {
        // 回归：旧缓冲以多字节字符结尾时，search_start = old_len - 7 可能落在
        // UTF-8 字符内部，直接 `&buffer[search_start..]` 会 panic（真实崩溃：
        // 预填 </think> 路径 + 中文推理链）。floor_char_boundary 修正到字符边界后，
        // 搜索区域略变大但不会漏检。
        let mut d = ContentThinkDemuxer::new();
        d.arm();
        // 8 个汉字 = 24 字节，末尾字符位于字节 21..24。
        assert_eq!(d.push("一二三四五六七八"), (String::new(), String::new()));
        // 第二次 push 后 search_start = 24 - 7 = 17，落在「六」(15..18) 内部。
        assert_eq!(
            d.push("</think>abcd"),
            ("一二三四五六七八".to_string(), "abcd".to_string())
        );
    }
}
