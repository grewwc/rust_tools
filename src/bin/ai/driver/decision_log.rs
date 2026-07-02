/// 决策日志模块 - 记录 AI Agent 的关键决策过程
///
/// 用于元认知（Meta-Cognition）：追溯"为什么做了某个选择"，便于调试和优化
use chrono::Local;
use rust_tools::cw::SkipMap;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

/// 磁盘决策日志的字节上限；超过即触发一次保留尾部的压缩。约 8MB。
const DECISION_LOG_MAX_PERSIST_BYTES: u64 = 8 * 1024 * 1024;
/// 压缩后保留的最近行数（与内存 max_capacity 同量级，足够回放当前会话）。
const DECISION_LOG_RETAIN_LINES: usize = 2000;

/// 决策类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DecisionType {
    /// 技能选择
    SkillSelection,
    /// 工具调用
    ToolInvocation,
    /// 模型路由（选择哪个模型）
    ModelRouting,
    /// Memory 检索
    MemoryRetrieval,
    /// Memory 保存门禁
    MemorySave,
    /// 反思触发
    ReflectionTrigger,
    /// 调度器分发与评估
    SchedulerDispatch,
}

/// 决策记录
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLog {
    /// 时间戳 (Unix timestamp in milliseconds)
    pub timestamp: i64,
    /// 会话 ID
    pub session_id: String,
    /// 轮次 ID
    pub turn_id: usize,
    /// 决策类型
    pub decision_type: DecisionType,
    /// 上下文（用户输入/当前状态）
    pub context: String,
    /// 考虑的备选方案
    pub alternatives_considered: Vec<String>,
    /// 最终选择的方案
    pub chosen_option: String,
    /// 选择理由
    pub reasoning: String,
    /// 置信度 (0.0 - 1.0)
    pub confidence: Option<f64>,
    /// 事后结果（执行后填充）
    pub outcome: Option<Outcome>,
    /// 执行耗时（毫秒）
    pub execution_time_ms: Option<u64>,
}

/// 决策结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Outcome {
    pub success: bool,
    pub message: String,
    pub user_feedback: Option<UserFeedback>,
}

/// 用户反馈
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum UserFeedback {
    Positive,
    Negative,
    Neutral,
}

/// 决策日志存储器
pub struct DecisionLogStore {
    logs: Arc<Mutex<Vec<DecisionLog>>>,
    max_capacity: usize,
    persist_path: Arc<Mutex<Option<PathBuf>>>,
}

impl DecisionLogStore {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::with_capacity(max_capacity))),
            max_capacity,
            persist_path: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_persist_path<P: AsRef<Path>>(&self, path: P) {
        let mut guard = self.persist_path.lock().unwrap();
        *guard = Some(path.as_ref().to_path_buf());
    }

    pub fn clear_persist_path(&self) {
        let mut guard = self.persist_path.lock().unwrap();
        *guard = None;
    }

    fn persist_log_if_enabled(&self, log: &DecisionLog) {
        let path = {
            let guard = self.persist_path.lock().unwrap();
            guard.clone()
        };
        let Some(path) = path else {
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) else {
            return;
        };
        let Ok(line) = serde_json::to_string(log) else {
            return;
        };
        let _ = writeln!(file, "{}", line);
        drop(file);

        // 决策日志是 append-only JSONL：内存缓冲受 max_capacity 限制，但磁盘文件
        // 若不轮转会无限增长，且 `replay_recent_from_disk` 每次都全量逐行读取。
        // 这里用一次 O(1) 的 metadata 探测，仅当超过上限时才做一次保留尾部的压缩。
        if let Ok(meta) = fs::metadata(&path)
            && meta.len() > DECISION_LOG_MAX_PERSIST_BYTES
        {
            self.compact_persist_file(&path);
        }
    }

    /// 把磁盘日志文件压缩到最近 `DECISION_LOG_RETAIN_LINES` 行，使用临时文件
    /// + 原子 rename，避免读到半截内容。best-effort：任意一步失败即放弃，不影响
    /// 主流程。并发写入时遵循 last-writer-wins，最多丢失少量尚未压缩的尾行。
    fn compact_persist_file(&self, path: &Path) {
        let Ok(file) = std::fs::File::open(path) else {
            return;
        };
        let reader = BufReader::new(file);
        let mut lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        if lines.len() <= DECISION_LOG_RETAIN_LINES {
            return;
        }
        let start = lines.len() - DECISION_LOG_RETAIN_LINES;
        let retained = lines.split_off(start);

        let tmp_path = path.with_extension("jsonl.tmp");
        let Ok(mut tmp) = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&tmp_path)
        else {
            return;
        };
        for line in &retained {
            if writeln!(tmp, "{}", line).is_err() {
                let _ = fs::remove_file(&tmp_path);
                return;
            }
        }
        if tmp.flush().is_err() {
            let _ = fs::remove_file(&tmp_path);
            return;
        }
        drop(tmp);
        let _ = fs::rename(&tmp_path, path);
    }

    /// 记录一个决策
    pub fn log(&self, mut log: DecisionLog) {
        let mut logs = self.logs.lock().unwrap();

        // 设置时间戳
        log.timestamp = Local::now().timestamp_millis();

        // 如果超出容量，删除最旧的 10%
        if logs.len() >= self.max_capacity {
            let remove_count = self.max_capacity / 10;
            logs.drain(0..remove_count);
        }

        let persist_copy = log.clone();
        logs.push(log);
        drop(logs);
        self.persist_log_if_enabled(&persist_copy);
    }

    /// 获取最近的 N 条日志
    pub fn recent(&self, n: usize) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        let start = logs.len().saturating_sub(n);
        logs[start..].to_vec()
    }

    pub fn recent_by_session(&self, session_id: &str, n: usize) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        let filtered = logs
            .iter()
            .filter(|log| log.session_id == session_id)
            .cloned()
            .collect::<Vec<_>>();
        let start = filtered.len().saturating_sub(n);
        filtered[start..].to_vec()
    }

    pub fn replay_recent_from_disk(&self, session_id: &str, n: usize) -> Vec<DecisionLog> {
        let path = {
            let guard = self.persist_path.lock().unwrap();
            guard.clone()
        };
        let Some(path) = path else {
            return Vec::new();
        };
        let Ok(file) = std::fs::File::open(path) else {
            return Vec::new();
        };
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(log) = serde_json::from_str::<DecisionLog>(&line)
                && log.session_id == session_id
            {
                out.push(log);
            }
        }
        let start = out.len().saturating_sub(n);
        out[start..].to_vec()
    }

    /// 按类型筛选日志
    pub fn by_type(&self, decision_type: &DecisionType) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| &log.decision_type == decision_type)
            .cloned()
            .collect()
    }

    /// 获取失败的决策日志
    pub fn failures(&self) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| log.outcome.as_ref().map(|o| !o.success).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// 获取低置信度的决策日志
    pub fn low_confidence(&self, threshold: f64) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| log.confidence.map(|c| c < threshold).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// 更新某个决策的结果
    pub fn update_outcome(&self, session_id: &str, turn_id: usize, outcome: Outcome) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs
            .iter_mut()
            .find(|log| log.session_id == session_id && log.turn_id == turn_id)
        {
            log.outcome = Some(outcome);
        }
    }

    /// 记录用户反馈
    pub fn add_feedback(&self, session_id: &str, turn_id: usize, feedback: UserFeedback) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs
            .iter_mut()
            .find(|log| log.session_id == session_id && log.turn_id == turn_id)
        {
            if let Some(outcome) = &mut log.outcome {
                outcome.user_feedback = Some(feedback);
            } else {
                log.outcome = Some(Outcome {
                    success: feedback != UserFeedback::Negative,
                    message: String::new(),
                    user_feedback: Some(feedback),
                });
            }
        }
    }

    /// 导出为 JSON 字符串
    pub fn export_json(&self, n: Option<usize>) -> String {
        let logs = self.logs.lock().unwrap();
        let logs_to_export = if let Some(n) = n {
            let start = logs.len().saturating_sub(n);
            &logs[start..]
        } else {
            &logs[..]
        };

        serde_json::to_string_pretty(logs_to_export)
            .unwrap_or_else(|e| format!("Error serializing logs: {}", e))
    }

    /// 统计信息
    pub fn stats(&self) -> DecisionStats {
        let logs = self.logs.lock().unwrap();

        let total = logs.len();
        let successes = logs
            .iter()
            .filter(|log| log.outcome.as_ref().map(|o| o.success).unwrap_or(false))
            .count();
        let failures = total - successes;

        let by_type: SkipMap<String, usize> = logs
            .iter()
            .map(|log| format!("{:?}", log.decision_type))
            .fold(SkipMap::default(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });

        let confidence_count = logs.iter().filter(|log| log.confidence.is_some()).count();
        let avg_confidence = if confidence_count > 0 {
            logs.iter().filter_map(|log| log.confidence).sum::<f64>() / (confidence_count as f64)
        } else {
            0.0
        };

        let exec_time_count = logs
            .iter()
            .filter(|log| log.execution_time_ms.is_some())
            .count();
        let avg_execution_time_ms = if exec_time_count > 0 {
            logs.iter()
                .filter_map(|log| log.execution_time_ms)
                .sum::<u64>() as f64
                / (exec_time_count as f64)
        } else {
            0.0
        };

        DecisionStats {
            total,
            successes,
            failures,
            success_rate: if total > 0 {
                successes as f64 / total as f64
            } else {
                0.0
            },
            by_type,
            avg_confidence,
            avg_execution_time_ms,
        }
    }

    /// 清空日志
    pub fn clear(&self) {
        let mut logs = self.logs.lock().unwrap();
        logs.clear();
    }
}

/// 决策统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionStats {
    pub total: usize,
    pub successes: usize,
    pub failures: usize,
    pub success_rate: f64,
    pub by_type: SkipMap<String, usize>,
    pub avg_confidence: f64,
    pub avg_execution_time_ms: f64,
}

/// 辅助函数：创建技能选择日志
pub fn log_skill_selection(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    candidates: Vec<&str>,
    chosen: &str,
    reasoning: &str,
    confidence: Option<f64>,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::SkillSelection,
        context: context.to_string(),
        alternatives_considered: candidates.iter().map(|s| s.to_string()).collect(),
        chosen_option: chosen.to_string(),
        reasoning: reasoning.to_string(),
        confidence,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

/// 辅助函数：创建工具调用日志
pub fn log_tool_invocation(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    tool_name: &str,
    reasoning: &str,
    confidence: Option<f64>,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::ToolInvocation,
        context: context.to_string(),
        alternatives_considered: vec![],
        chosen_option: tool_name.to_string(),
        reasoning: reasoning.to_string(),
        confidence,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

/// 辅助函数：创建 Memory 检索日志
pub fn log_memory_retrieval(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    query: &str,
    results_count: usize,
    reasoning: &str,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::MemoryRetrieval,
        context: query.to_string(),
        alternatives_considered: vec![],
        chosen_option: format!("Retrieved {} items", results_count),
        reasoning: reasoning.to_string(),
        confidence: None,
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

pub fn log_memory_save_assessment(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    requested_category: &str,
    final_category: &str,
    note: &str,
    assessment: &crate::ai::driver::reflection::LearningNoteAssessment,
    downgraded: bool,
) {
    let note_chars = note.chars().count();
    let preview: String = note.chars().take(160).collect();
    let context = serde_json::json!({
        "requested_category": requested_category,
        "final_category": final_category,
        "downgraded": downgraded,
        "note_chars": note_chars,
        "note_preview": preview,
    })
    .to_string();
    let reasoning = serde_json::to_string(assessment).unwrap_or_else(|_| "{}".to_string());
    let outcome_message = if downgraded {
        "memory_save downgraded to short-term self_note"
    } else {
        "memory_save accepted for requested category"
    };
    store.log(DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::MemorySave,
        context,
        alternatives_considered: vec![requested_category.to_string(), "self_note".to_string()],
        chosen_option: final_category.to_string(),
        reasoning,
        confidence: Some(assessment.confidence()),
        outcome: Some(Outcome {
            success: !downgraded,
            message: outcome_message.to_string(),
            user_feedback: None,
        }),
        execution_time_ms: None,
    });
}

/// 辅助函数：记录调度器分发决策（含 defer/selected 与评分摘要）
pub fn log_scheduler_dispatch(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    context: &str,
    alternatives: Vec<String>,
    chosen: &str,
    reasoning: &str,
    success: bool,
) {
    store.log(DecisionLog {
        timestamp: 0,
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::SchedulerDispatch,
        context: context.to_string(),
        alternatives_considered: alternatives,
        chosen_option: chosen.to_string(),
        reasoning: reasoning.to_string(),
        confidence: None,
        outcome: Some(Outcome {
            success,
            message: if success {
                "scheduler decision accepted".to_string()
            } else {
                "scheduler decision indicates risk".to_string()
            },
            user_feedback: None,
        }),
        execution_time_ms: None,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_log_path(name: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        path.push(format!(
            "rust_tools_{name}_{}_{}.jsonl",
            std::process::id(),
            ts
        ));
        path
    }

    #[test]
    fn test_log_store_basic() {
        let store = DecisionLogStore::new(100);

        store.log(DecisionLog {
            timestamp: 0,
            session_id: "test-session".to_string(),
            turn_id: 1,
            decision_type: DecisionType::SkillSelection,
            context: "test input".to_string(),
            alternatives_considered: vec!["skill_a".to_string(), "skill_b".to_string()],
            chosen_option: "skill_a".to_string(),
            reasoning: "test reasoning".to_string(),
            confidence: Some(0.85),
            outcome: None,
            execution_time_ms: Some(10),
        });

        let recent = store.recent(10);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].chosen_option, "skill_a");
    }

    #[test]
    fn test_log_store_capacity() {
        let store = DecisionLogStore::new(10);

        // 添加 15 条日志
        for i in 0..15 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::SkillSelection,
                context: format!("input {}", i),
                alternatives_considered: vec![],
                chosen_option: format!("skill_{}", i),
                reasoning: "test".to_string(),
                confidence: None,
                outcome: None,
                execution_time_ms: None,
            });
        }

        // 应该只保留最近的 10 条（实际上会保留 9-10 条，因为会删除 10%）
        let recent = store.recent(100);
        assert!(recent.len() <= 10);
        assert_eq!(recent[0].turn_id, 5); // 最旧的是第 5 条
    }

    #[test]
    fn test_outcome_update() {
        let store = DecisionLogStore::new(100);

        store.log(DecisionLog {
            timestamp: 0,
            session_id: "test-session".to_string(),
            turn_id: 1,
            decision_type: DecisionType::ToolInvocation,
            context: "test".to_string(),
            alternatives_considered: vec![],
            chosen_option: "tool_x".to_string(),
            reasoning: "test".to_string(),
            confidence: None,
            outcome: None,
            execution_time_ms: None,
        });

        store.update_outcome(
            "test-session",
            1,
            Outcome {
                success: true,
                message: "Tool executed successfully".to_string(),
                user_feedback: None,
            },
        );

        let recent = store.recent(1);
        assert!(recent[0].outcome.as_ref().unwrap().success);
    }

    #[test]
    fn test_stats() {
        let store = DecisionLogStore::new(100);

        // 添加成功和失败的日志
        for i in 0..5 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::SkillSelection,
                context: "test".to_string(),
                alternatives_considered: vec![],
                chosen_option: format!("skill_{}", i),
                reasoning: "test".to_string(),
                confidence: Some(0.8),
                outcome: Some(Outcome {
                    success: true,
                    message: "OK".to_string(),
                    user_feedback: None,
                }),
                execution_time_ms: Some(10),
            });
        }

        for i in 5..10 {
            store.log(DecisionLog {
                timestamp: 0,
                session_id: "test-session".to_string(),
                turn_id: i,
                decision_type: DecisionType::ToolInvocation,
                context: "test".to_string(),
                alternatives_considered: vec![],
                chosen_option: format!("tool_{}", i),
                reasoning: "test".to_string(),
                confidence: Some(0.6),
                outcome: Some(Outcome {
                    success: false,
                    message: "Failed".to_string(),
                    user_feedback: None,
                }),
                execution_time_ms: Some(20),
            });
        }

        let stats = store.stats();
        assert_eq!(stats.total, 10);
        assert_eq!(stats.successes, 5);
        assert_eq!(stats.failures, 5);
        assert!((stats.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_persist_and_replay_recent_by_session() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_persist");
        store.set_persist_path(&path);

        for turn in 0..5usize {
            log_scheduler_dispatch(
                &store,
                "sess-a",
                turn,
                "ctx",
                vec!["a".to_string()],
                "chosen",
                "reason",
                true,
            );
        }
        for turn in 0..3usize {
            log_scheduler_dispatch(
                &store,
                "sess-b",
                turn,
                "ctx",
                vec!["b".to_string()],
                "chosen",
                "reason",
                false,
            );
        }

        let replay = store.replay_recent_from_disk("sess-a", 3);
        assert_eq!(replay.len(), 3);
        assert!(replay.iter().all(|item| item.session_id == "sess-a"));

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_clear_persist_path_disables_disk_write() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_disabled");
        store.set_persist_path(&path);
        store.clear_persist_path();

        log_scheduler_dispatch(
            &store,
            "sess-a",
            0,
            "ctx",
            vec!["a".to_string()],
            "chosen",
            "reason",
            true,
        );

        assert!(!path.exists());
    }

    #[test]
    fn test_compact_persist_file_retains_recent_tail() {
        let store = DecisionLogStore::new(100);
        let path = temp_log_path("decision_log_compact");

        // 直接写入超过保留上限的行数，再手动触发压缩。
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let total = DECISION_LOG_RETAIN_LINES + 500;
        {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap();
            for i in 0..total {
                let log = DecisionLog {
                    timestamp: i as i64,
                    session_id: "sess".to_string(),
                    turn_id: i,
                    decision_type: DecisionType::SchedulerDispatch,
                    context: "ctx".to_string(),
                    alternatives_considered: vec![],
                    chosen_option: "c".to_string(),
                    reasoning: "r".to_string(),
                    confidence: None,
                    outcome: None,
                    execution_time_ms: None,
                };
                writeln!(file, "{}", serde_json::to_string(&log).unwrap()).unwrap();
            }
        }

        store.compact_persist_file(&path);

        let reader = BufReader::new(std::fs::File::open(&path).unwrap());
        let lines: Vec<String> = reader.lines().map_while(Result::ok).collect();
        assert_eq!(lines.len(), DECISION_LOG_RETAIN_LINES);
        // 应保留最新的尾部：最后一行 turn_id == total - 1。
        let last: DecisionLog = serde_json::from_str(lines.last().unwrap()).unwrap();
        assert_eq!(last.turn_id, total - 1);
        // 最旧保留行应为 total - RETAIN_LINES。
        let first: DecisionLog = serde_json::from_str(&lines[0]).unwrap();
        assert_eq!(first.turn_id, total - DECISION_LOG_RETAIN_LINES);

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_log_memory_save_assessment_records_structured_reasoning() {
        let store = DecisionLogStore::new(100);
        let assessment = crate::ai::driver::reflection::LearningNoteAssessment {
            actionable: false,
            specific: false,
            generalizable: false,
            score: 0,
            high_quality: false,
            char_count: 10,
            word_count: 2,
            nonempty_lines: 1,
            unique_token_ratio: 1.0,
            directive_signals: 0,
            code_signals: 0,
            artifact_signals: 0,
            abstraction_signals: 0,
            condition_signals: 0,
            one_off_signals: 0,
        };

        log_memory_save_assessment(
            &store,
            "sess-test",
            7,
            "common_sense",
            "self_note",
            "be careful",
            &assessment,
            true,
        );

        let recent = store.recent(1);
        assert_eq!(recent[0].decision_type, DecisionType::MemorySave);
        assert_eq!(recent[0].chosen_option, "self_note");
        assert!(recent[0].reasoning.contains("\"score\":0"));
        assert!(recent[0].context.contains("requested_category"));
    }
}

// 全局单例访问
use std::sync::OnceLock;

static DECISION_LOG_STORE: OnceLock<DecisionLogStore> = OnceLock::new();

/// 获取全局决策日志存储
pub fn get_decision_log_store() -> &'static DecisionLogStore {
    DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(1000))
}

/// 初始化决策日志存储（可选，用于自定义容量）
pub fn init_decision_log_store(capacity: usize) -> &'static DecisionLogStore {
    DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(capacity))
}

pub fn init_decision_log_store_with_path<P: AsRef<Path>>(
    capacity: usize,
    path: P,
) -> &'static DecisionLogStore {
    let store = DECISION_LOG_STORE.get_or_init(|| DecisionLogStore::new(capacity));
    store.set_persist_path(path);
    store
}

pub fn set_decision_log_persist_path<P: AsRef<Path>>(path: P) {
    get_decision_log_store().set_persist_path(path);
}

pub fn clear_decision_log_persist_path() {
    get_decision_log_store().clear_persist_path();
}
