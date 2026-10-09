//! Conservative budgeting for the two-message, tool-free distillation request.

use std::ops::Range;

use serde::Serialize;
use serde_json::{Value, json};

use crate::ai::models;

use super::MODEL_RULES;

// Reserve space for protocol-specific fields, reasoning controls, and message
// framing beyond the serialized chat envelope. Payload bytes are charged at one
// token per UTF-8 byte, including both layers of JSON escaping.
const ENVELOPE_RESERVE_TOKENS: usize = 4096;
// Bound speculative serialization independently of the archive/catalog length.
const MAX_PAGE_ITEMS: usize = 64;

#[derive(Clone, Debug)]
pub(super) struct RequestBudget {
    request_model: String,
    input_bytes: usize,
}

impl RequestBudget {
    pub(super) fn for_model(model: &str) -> Result<Self, String> {
        // Tier-based context defaults are estimates, not declared model limits.
        let context = crate::ai::model_names::find_by_identifier(model)
            .and_then(|definition| definition.context_window_tokens)
            .filter(|tokens| *tokens > 0)
            .ok_or_else(|| format!("distill model {model:?} has no declared context window"))?;
        // do_request_text_streaming does not send max_tokens or honor the CLI
        // override. Reserve the full declared output maximum, not the main chat
        // builder's dynamically clamped allowance. Unknown provider defaults
        // cannot safely be replaced by a guessed output allowance.
        let output = models::max_output_tokens(model)
            .filter(|tokens| *tokens > 0)
            .ok_or_else(|| format!("distill model {model:?} has no declared output maximum"))?;
        Self::from_limits(
            models::request_model_name(model).to_string(),
            context,
            output as usize,
        )
    }

    fn from_limits(request_model: String, context: usize, output: usize) -> Result<Self, String> {
        let input_bytes = context
            .checked_sub(output)
            .and_then(|remaining| remaining.checked_sub(ENVELOPE_RESERVE_TOKENS))
            .filter(|remaining| *remaining > 0)
            .ok_or_else(|| {
                format!(
                    "distill model {request_model:?} has no input budget: context={context}, output={output}, envelope_reserve={ENVELOPE_RESERVE_TOKENS}"
                )
            })?;
        let budget = Self {
            request_model,
            input_bytes,
        };
        budget.check(&json!({}))?;
        Ok(budget)
    }

    /// Inject the complete serialized input-byte capacity after output/framing
    /// reserves; MODEL_RULES and both JSON envelopes still consume this budget.
    #[cfg(test)]
    pub(super) fn for_test(input_bytes: usize) -> Self {
        Self {
            request_model: "session-distill-test".to_string(),
            input_bytes,
        }
    }

    fn request_bytes(&self, payload: &Value) -> usize {
        let envelope = json!({
            "model": self.request_model,
            "messages": [
                {"role": "system", "content": MODEL_RULES},
                {"role": "user", "content": payload.to_string()},
            ],
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        envelope.to_string().len()
    }

    pub(super) fn fits(&self, payload: &Value) -> bool {
        self.request_bytes(payload) <= self.input_bytes
    }

    pub(super) fn check(&self, payload: &Value) -> Result<(), String> {
        let needed = self.request_bytes(payload);
        if needed <= self.input_bytes {
            Ok(())
        } else {
            Err(format!(
                "distill request requires {needed} serialized input bytes, exceeding the {}-byte budget (one byte per token; output and framing reserved separately)",
                self.input_bytes
            ))
        }
    }

    /// Return ordered, nonempty ranges covering every item, or fail explicitly
    /// when even a singleton's complete request cannot fit. The payload factory
    /// must include all fixed fields and only the supplied page's variable items.
    pub(super) fn pages<T: Serialize>(
        &self,
        items: &[T],
        payload: impl Fn(&[T]) -> Value,
    ) -> Result<Vec<Range<usize>>, String> {
        let mut pages = Vec::new();
        let mut start = 0;
        while start < items.len() {
            let mut end = start + 1;
            self.check(&payload(&items[start..end])).map_err(|error| {
                format!("distill single item at index {start} is oversized: {error}")
            })?;
            let limit = start.saturating_add(MAX_PAGE_ITEMS).min(items.len());
            while end < limit {
                let probe = start + ((end - start) * 2).min(limit - start);
                if self.fits(&payload(&items[start..probe])) {
                    end = probe;
                    continue;
                }
                // Both endpoints are known: end fits and probe does not.
                let mut high = probe;
                while high - end > 1 {
                    let middle = end + (high - end) / 2;
                    if self.fits(&payload(&items[start..middle])) {
                        end = middle;
                    } else {
                        high = middle;
                    }
                }
                break;
            }
            pages.push(start..end);
            start = end;
        }
        Ok(pages)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    #[test]
    fn session_distill_budget_counts_rules_envelope_and_json_escaping() {
        let payload = json!({"source": "é漢🙂\n\"\\\u{0}"});
        let unlimited = RequestBudget::for_test(usize::MAX);
        let needed = unlimited.request_bytes(&payload);
        assert!(needed > MODEL_RULES.len() + payload.to_string().len());
        let exact = RequestBudget::for_test(needed);
        assert!(exact.fits(&payload));
        assert!(exact.check(&payload).is_ok());
        assert!(!RequestBudget::for_test(needed - 1).fits(&payload));
        assert!(RequestBudget::for_test(needed - 1).check(&payload).is_err());
    }

    #[test]
    fn session_distill_budget_reserves_full_output_and_framing() {
        let input = 16_000;
        let output = 8_000;
        let context = input + output + ENVELOPE_RESERVE_TOKENS;
        let budget = RequestBudget::from_limits("test".into(), context, output).unwrap();
        assert_eq!(budget.input_bytes, input);
        assert!(RequestBudget::from_limits("test".into(), output - 1, output).is_err());
        assert!(
            RequestBudget::from_limits("test".into(), output + ENVELOPE_RESERVE_TOKENS, output)
                .is_err()
        );
    }

    #[test]
    fn session_distill_budget_empty_does_not_build_payload() {
        let pages = RequestBudget::for_test(0)
            .pages::<usize>(&[], |_| panic!("empty input must not be serialized"))
            .unwrap();
        assert!(pages.is_empty());
    }

    #[test]
    fn session_distill_budget_oversized_single_item_is_explicit() {
        let items = vec!["small".to_string(), "x".repeat(20_000)];
        let payload = |page: &[String]| json!({"items": page});
        let capacity = RequestBudget::for_test(usize::MAX).request_bytes(&payload(&items[..1]));
        let error = RequestBudget::for_test(capacity)
            .pages(&items, payload)
            .unwrap_err();
        assert!(
            error.contains("single item at index 1 is oversized"),
            "{error}"
        );
    }

    #[test]
    fn session_distill_budget_pages_cover_all_items_with_fixed_fields() {
        let items = vec!["x".repeat(128); 101];
        let payload = |page: &[String]| json!({"instructions": "fixed".repeat(80), "items": page});
        let capacity = RequestBudget::for_test(usize::MAX).request_bytes(&payload(&items[..3]));
        let budget = RequestBudget::for_test(capacity);
        let pages = budget.pages(&items, payload).unwrap();
        let mut next = 0;
        for page in pages {
            assert_eq!(page.start, next);
            assert!(!page.is_empty());
            assert!(page.len() <= 3);
            assert!(budget.fits(&payload(&items[page.clone()])));
            next = page.end;
        }
        assert_eq!(next, items.len());
    }

    #[test]
    fn session_distill_budget_bounds_speculative_serialization() {
        let items = vec![0; MAX_PAGE_ITEMS * 20 + 3];
        let largest = Cell::new(0);
        let pages = RequestBudget::for_test(usize::MAX)
            .pages(&items, |page| {
                largest.set(largest.get().max(page.len()));
                assert!(page.len() <= MAX_PAGE_ITEMS);
                json!({"items": page})
            })
            .unwrap();
        assert_eq!(largest.get(), MAX_PAGE_ITEMS);
        assert_eq!(
            pages.iter().map(|page| page.len()).sum::<usize>(),
            items.len()
        );
        assert_eq!(pages.last().unwrap().end, items.len());
    }
}
