//! LLM 请求 TPM 预检限速。
//!
//! 429 的根因不是单次请求超上下文，而是同一 turn 内连续请求把 prompt+tool schema
//! 在 60 秒窗口里反复发送。这里在每次 physical HTTP send 前做滑动窗口预算预占：
//! 超预算就可取消等待，预算释放后再发送。这样不裁剪工具、不硬砍迭代，只控制发送速率。

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

use rustc_hash::FxHashMap;
use serde_json::Value;

use crate::ai::{models, types::App};

use super::error::{RequestError, sleep_with_cancel};

const TPM_WINDOW: Duration = Duration::from_secs(60);

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

fn configured_tpm_limit(model: &str) -> Option<u64> {
    models::request_tpm_limit(model)
}

fn reservation_tokens(estimated_prompt_tokens: usize, physical_sends: usize) -> u64 {
    let estimate = u64::try_from(estimated_prompt_tokens).unwrap_or(u64::MAX / 2);
    let sends = u64::try_from(physical_sends.max(1)).unwrap_or(u64::MAX / 2);
    estimate.saturating_mul(sends).max(1)
}

/// 用服务端上一轮返回的真实 prompt_tokens 校准字符估算。
///
/// 字符估算按 2 chars/token 折算，对英文代码和 JSON schema 会显著高估；截图里的
/// 真实案例是服务端报告 25,875 prompt tokens，而字符估算路径预占 55,610。
/// 这种高估会让普通工具循环过早 sleep，表现为"还没真正接近 TPM 就卡住"。
/// 若上一轮还有 `cached_tokens`，则预算优先按"未缓存尾巴 + 本轮新增部分"估算，
/// 避免 prompt cache 100% hit 时仍按整段 prompt 预占。
///
/// 但上一轮 usage 也可能因为本轮新增大工具结果而偏低，因此不直接全信 known：
/// - known 低于字符估算一半时，按字符估算的一半作为地板；
/// - known 高于字符估算时，按字符估算作为上界，避免历史压缩后沿用旧高值。
pub(super) fn calibrate_prompt_tokens_for_budget(
    estimated_prompt_tokens: usize,
    known_prompt_tokens: Option<u64>,
    known_cached_prompt_tokens: Option<u64>,
) -> usize {
    let Some(known) = known_prompt_tokens.and_then(|v| usize::try_from(v).ok()) else {
        return estimated_prompt_tokens.max(1);
    };
    if estimated_prompt_tokens == 0 {
        return known.max(1);
    }
    let known_cached = known_cached_prompt_tokens
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(0)
        .min(known);
    if known_cached > 0 {
        let known_uncached = known.saturating_sub(known_cached).max(1);
        let reusable_cache = known_cached.min(estimated_prompt_tokens);
        return estimated_prompt_tokens
            .saturating_sub(reusable_cache)
            .max(known_uncached);
    }
    let floor = estimated_prompt_tokens.div_ceil(2).max(1);
    known.clamp(floor, estimated_prompt_tokens.max(floor))
}

fn budget_key(endpoint: &str, request_model: &str, api_key: &str) -> String {
    let mut hasher = rustc_hash::FxHasher::default();
    api_key.trim().hash(&mut hasher);
    let key_fp = hasher.finish();
    format!(
        "{}|{}|{:016x}",
        endpoint.trim(),
        request_model.trim(),
        key_fp
    )
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
                self.reservations
                    .push_back(TokenReservation { at: now, tokens });
                return BudgetDecision::Reserved;
            }
            return BudgetDecision::Wait(self.next_release_delay(now, window));
        }

        if used.saturating_add(tokens) <= limit {
            self.reservations
                .push_back(TokenReservation { at: now, tokens });
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
    model: &str,
    endpoint: &str,
    request_model_label: &str,
    api_key: &str,
    estimated_prompt_tokens: usize,
    physical_sends: usize,
) -> Result<(), RequestError> {
    let Some(limit) = configured_tpm_limit(model) else {
        return Ok(());
    };
    let tokens = reservation_tokens(estimated_prompt_tokens, physical_sends);
    let key = budget_key(endpoint, request_model_label, api_key);

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
                    "[Info] request TPM budget reached for `{request_model_label}`; waiting {:.1}s before next send (reserved {} / model limit {} tokens, key-scoped 60s window)",
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
pub(super) fn test_reservation_tokens(
    estimated_prompt_tokens: usize,
    physical_sends: usize,
) -> u64 {
    reservation_tokens(estimated_prompt_tokens, physical_sends)
}

#[cfg(test)]
pub(super) fn test_budget_key(endpoint: &str, request_model: &str, api_key: &str) -> String {
    budget_key(endpoint, request_model, api_key)
}
