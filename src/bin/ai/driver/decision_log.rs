/// 决策日志模块 - 记录 AI Agent 的关键决策过程
/// 
/// 用于元认知（Meta-Cognition）：追溯"为什么做了某个选择"，便于调试和优化

use chrono::Local;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

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
    /// 反思触发
    ReflectionTrigger,
    /// 用户意图识别
    IntentRecognition,
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
}

impl DecisionLogStore {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::with_capacity(max_capacity))),
            max_capacity,
        }
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
        
        logs.push(log);
    }

    /// 获取最近的 N 条日志
    pub fn recent(&self, n: usize) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        let start = logs.len().saturating_sub(n);
        logs[start..].to_vec()
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
            .filter(|log| {
                log.outcome.as_ref().map(|o| !o.success).unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// 获取低置信度的决策日志
    pub fn low_confidence(&self, threshold: f64) -> Vec<DecisionLog> {
        let logs = self.logs.lock().unwrap();
        logs.iter()
            .filter(|log| {
                log.confidence.map(|c| c < threshold).unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// 更新某个决策的结果
    pub fn update_outcome(
        &self,
        session_id: &str,
        turn_id: usize,
        outcome: Outcome,
    ) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs.iter_mut().find(|log| {
            log.session_id == session_id && log.turn_id == turn_id
        }) {
            log.outcome = Some(outcome);
        }
    }

    /// 记录用户反馈
    pub fn add_feedback(
        &self,
        session_id: &str,
        turn_id: usize,
        feedback: UserFeedback,
    ) {
        let mut logs = self.logs.lock().unwrap();
        if let Some(log) = logs.iter_mut().find(|log| {
            log.session_id == session_id && log.turn_id == turn_id
        }) {
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
        
        serde_json::to_string_pretty(logs_to_export).unwrap_or_else(|e| {
            format!("Error serializing logs: {}", e)
        })
    }

    /// 统计信息
    pub fn stats(&self) -> DecisionStats {
        let logs = self.logs.lock().unwrap();
        
        let total = logs.len();
        let successes = logs.iter().filter(|log| {
            log.outcome.as_ref().map(|o| o.success).unwrap_or(false)
        }).count();
        let failures = total - successes;
        
        let by_type: std::collections::HashMap<String, usize> = logs
            .iter()
            .map(|log| format!("{:?}", log.decision_type))
            .fold(std::collections::HashMap::new(), |mut acc, t| {
                *acc.entry(t).or_insert(0) += 1;
                acc
            });
        
        let confidence_count = logs.iter().filter(|log| log.confidence.is_some()).count();
        let avg_confidence = if confidence_count > 0 {
            logs.iter()
                .filter_map(|log| log.confidence)
                .sum::<f64>() / (confidence_count as f64)
        } else {
            0.0
        };
        
        let exec_time_count = logs.iter().filter(|log| log.execution_time_ms.is_some()).count();
        let avg_execution_time_ms = if exec_time_count > 0 {
            logs.iter()
                .filter_map(|log| log.execution_time_ms)
                .sum::<u64>() as f64 / (exec_time_count as f64)
        } else {
            0.0
        };
        
        DecisionStats {
            total,
            successes,
            failures,
            success_rate: if total > 0 { successes as f64 / total as f64 } else { 0.0 },
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
    pub by_type: std::collections::HashMap<String, usize>,
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

/// 辅助函数：创建意图识别日志
pub fn log_intent_recognition(
    store: &DecisionLogStore,
    session_id: &str,
    turn_id: usize,
    input: &str,
    detected_intent: &str,
    alternatives: Vec<&str>,
    confidence: f64,
    execution_time_ms: u64,
) {
    store.log(DecisionLog {
        timestamp: 0, // Will be set by log()
        session_id: session_id.to_string(),
        turn_id,
        decision_type: DecisionType::IntentRecognition,
        context: input.to_string(),
        alternatives_considered: alternatives.iter().map(|s| s.to_string()).collect(),
        chosen_option: detected_intent.to_string(),
        reasoning: format!("Confidence: {:.2}", confidence),
        confidence: Some(confidence),
        outcome: None,
        execution_time_ms: Some(execution_time_ms),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
