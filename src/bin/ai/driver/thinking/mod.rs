mod engine;
mod verification;
mod generalization;
mod goals;
mod orchestrator;

pub use orchestrator::ThinkingOrchestrator;

use std::sync::{Mutex, OnceLock};

/// 全局 raw_experience 入口。供 tool 失败路径等无 orchestrator 上下文的调用方
/// 把"经验"直接落到持久化 store，由后续 orchestrator 启动时统一被泛化引擎消费。
///
/// 使用一个独立的进程级 Mutex<ExperienceGeneralizer>，仅用于做 ingest（实际只
/// 触发 persist + buffer push）。不与 ThinkingOrchestrator 内的 generalizer 共
/// 享内存状态——每个 turn 的 orchestrator 在构造时会从 store 重新加载 buffer，
/// 所以全局 ingest 写到 store 即可被后续 turn 读到。
static GLOBAL_INGEST: OnceLock<Mutex<generalization::ExperienceGeneralizer>> = OnceLock::new();

pub(crate) fn ingest_raw_experience_global(
    category: &str,
    note: &str,
    tags: &[String],
    source: Option<&str>,
) {
    let m = GLOBAL_INGEST.get_or_init(|| Mutex::new(generalization::ExperienceGeneralizer::new()));
    if let Ok(mut g) = m.lock() {
        g.ingest_experience(category, note, tags, source);
    }
}
