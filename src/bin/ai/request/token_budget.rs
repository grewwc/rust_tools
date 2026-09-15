//! TPM preflight rate limiting for LLM requests.
//!
//! The 429 issue addressed here comes from repeatedly sending prompt + tool schemas within a
//! 60-second window during one turn, not a single request exceeding context. Reserve sliding-window
//! budget before each physical HTTP send; wait cancellably when exhausted, limiting rate without cutting tools or iterations.

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

/// Convert a current-request context estimate into a TPM reservation. The caller
/// must validate cache reuse against the originating request and API key. Cache
/// reuse is a rate-accounting heuristic and never reduces context occupancy.
pub(super) fn tpm_prompt_tokens(
    context_prompt_tokens: usize,
    reusable_cached_prompt_tokens: Option<u64>,
) -> usize {
    let cached = reusable_cached_prompt_tokens
        .and_then(|tokens| usize::try_from(tokens).ok())
        .unwrap_or(0);
    context_prompt_tokens.saturating_sub(cached).max(1)
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

        // If one request reaches the window limit, wait for old reservations to expire, then admit it to avoid starvation.
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

/// Estimates tokens from serialized bytes: a byte-input variant of `estimate_json_request_tokens`
/// that uses the existing serialized data rather than serializing the entire request body again.
pub(super) fn estimate_serialized_request_tokens(bytes: &[u8]) -> usize {
    const CHARS_PER_TOKEN_CONSERVATIVE: usize = 2;
    std::str::from_utf8(bytes)
        .map(|s| s.chars().count().div_ceil(CHARS_PER_TOKEN_CONSERVATIVE))
        .unwrap_or_else(|_| bytes.len().div_ceil(CHARS_PER_TOKEN_CONSERVATIVE))
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

    let mut status_line = super::TransientStatusLine::new();
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
                let msg = format!(
                    "⌛ TPM 限流 · `{request_model_label}` · 等待 {:.1}s (预占 {} / 限额 {} tokens, 60s 窗口)",
                    delay.as_secs_f32(),
                    tokens,
                    limit
                );
                if let Some(line) = status_line.as_mut() {
                    line.update(&msg);
                } else {
                    super::emit_request_diagnostic(format_args!("{msg}"));
                }
                if sleep_with_cancel(app, delay).await {
                    // Clear the transient status line on cancellation so it does not linger.
                    drop(status_line.take());
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
