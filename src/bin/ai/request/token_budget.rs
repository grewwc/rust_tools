//! LLM 请求 TPM 预检限速。
//!
//! 429 的根因不是单次请求超上下文，而是同一 turn 内连续请求把 prompt+tool schema
//! 在 60 秒窗口里反复发送。这里在每次 physical HTTP send 前做滑动窗口预算预占：
//! 超预算就可取消等待，预算释放后再发送。这样不裁剪工具、不硬砍迭代，只控制发送速率。

use std::collections::VecDeque;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::ai::{config_schema::AiConfig, types::App};
use rust_tools::commonw::configw;

use super::error::{RequestError, sleep_with_cancel};

const DEFAULT_REQUEST_TPM_LIMIT: u64 = 380_000;
const TPM_WINDOW: Duration = Duration::from_secs(60);
/// 预占时给字符估算留 20% 安全余量，覆盖中英文比例、wire 模板、hedged backup 等
/// 估算误差。宁可稍微多等，也不要把请求打到 429 后再重试。
const ESTIMATE_SAFETY_NUMERATOR: u64 = 6;
const ESTIMATE_SAFETY_DENOMINATOR: u64 = 5;

#[derive(Clone, Copy)]
struct TokenReservation {
    at: Instant,
    tokens: u64,
}

#[derive(Default)]
pub(super) struct TokenBudgetBucket {
    reservations: VecDeque<TokenReservation>,
}

#[derive(Default)]
struct TokenBudgetState {
    buckets: FxHashMap<String, TokenBudgetBucket>,
}

pub(super) enum BudgetDecision {
    Reserved,
    Wait(Duration),
}

static STATE: LazyLock<Mutex<TokenBudgetState>> =
    LazyLock::new(|| Mutex::new(TokenBudgetState::default()));

fn configured_tpm_limit() -> Option<u64> {
    let raw = configw::get_all_config().get_opt(AiConfig::REQUEST_TPM_LIMIT);
    let Some(raw) = raw else {
        return Some(DEFAULT_REQUEST_TPM_LIMIT);
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Some(DEFAULT_REQUEST_TPM_LIMIT);
    }
    match trimmed.parse::<u64>() {
        Ok(0) => None,
        Ok(limit) => Some(limit),
        Err(_) => Some(DEFAULT_REQUEST_TPM_LIMIT),
    }
}

fn reservation_tokens(estimated_prompt_tokens: usize, physical_sends: usize) -> u64 {
    let estimate = u64::try_from(estimated_prompt_tokens).unwrap_or(u64::MAX / 2);
    let sends = u64::try_from(physical_sends.max(1)).unwrap_or(u64::MAX / 2);
    estimate
        .saturating_mul(ESTIMATE_SAFETY_NUMERATOR)
        .div_ceil(ESTIMATE_SAFETY_DENOMINATOR)
        .saturating_mul(sends)
        .max(1)
}

fn budget_key(endpoint: &str, request_model: &str) -> String {
    format!("{}|{}", endpoint.trim(), request_model.trim())
}

impl TokenBudgetBucket {
    fn prune(&mut self, now: Instant, window: Duration) {
        while self
            .reservations
            .front()
            .is_some_and(|entry| now.duration_since(entry.at) >= window)
        {
            self.reservations.pop_front();
        }
    }

    fn used_tokens(&self) -> u64 {
        self.reservations
            .iter()
            .fold(0u64, |acc, item| acc.saturating_add(item.tokens))
    }

    pub(super) fn reserve_or_delay(
        &mut self,
        now: Instant,
        limit: u64,
        tokens: u64,
        window: Duration,
    ) -> BudgetDecision {
        let limit = limit.max(1);
        self.prune(now, window);
        let used = self.used_tokens();

        // 单次请求估算已超过窗口时，等待旧账清空后放行，避免永远无法发送。
        if tokens >= limit {
            if used == 0 {
                self.reservations.push_back(TokenReservation { at: now, tokens });
                return BudgetDecision::Reserved;
            }
            return BudgetDecision::Wait(self.next_release_delay(now, window));
        }

        if used.saturating_add(tokens) <= limit {
            self.reservations.push_back(TokenReservation { at: now, tokens });
            BudgetDecision::Reserved
        } else {
            BudgetDecision::Wait(self.next_release_delay(now, window))
        }
    }

    fn next_release_delay(&self, now: Instant, window: Duration) -> Duration {
        self.reservations
            .front()
            .map(|entry| {
                entry
                    .at
                    .checked_add(window)
                    .and_then(|deadline| deadline.checked_duration_since(now))
                    .unwrap_or(Duration::from_millis(1))
            })
            .unwrap_or(Duration::from_millis(1))
            .max(Duration::from_millis(1))
    }
}

pub(super) fn estimate_json_request_tokens(value: &Value) -> usize {
    const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
    serde_json::to_string(value)
        .map(|s| s.chars().count().div_ceil(CHARS_PER_TOKEN_CONSERVATIVE))
        .unwrap_or(1)
        .max(1)
}

pub(super) async fn wait_for_request_budget(
    app: &App,
    endpoint: &str,
    request_model: &str,
    estimated_prompt_tokens: usize,
    physical_sends: usize,
) -> Result<(), RequestError> {
    let Some(limit) = configured_tpm_limit() else {
        return Ok(());
    };
    let tokens = reservation_tokens(estimated_prompt_tokens, physical_sends);
    let key = budget_key(endpoint, request_model);

    loop {
        let decision = {
            let Ok(mut state) = STATE.lock() else {
                return Ok(());
            };
            let bucket = state.buckets.entry(key.clone()).or_default();
            bucket.reserve_or_delay(Instant::now(), limit, tokens, TPM_WINDOW)
        };

        match decision {
            BudgetDecision::Reserved => return Ok(()),
            BudgetDecision::Wait(delay) => {
                eprintln!(
                    "[Info] request TPM budget reached for `{request_model}`; waiting {:.1}s before next send (reserved {} / limit {} tokens, key-scoped 60s window)",
                    delay.as_secs_f32(),
                    tokens,
                    limit
                );
                if sleep_with_cancel(app, delay).await {
                    return Err(RequestError::cancelled(
                        "request canceled by user during TPM budget wait",
                    ));
                }
            }
        }
    }
}

#[cfg(test)]
pub(super) fn test_reservation_tokens(estimated_prompt_tokens: usize, physical_sends: usize) -> u64 {
    reservation_tokens(estimated_prompt_tokens, physical_sends)
}
