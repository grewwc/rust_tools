use super::*;

impl LlmOps for LocalOS {
    fn llm_set_price(&mut self, model: String, price: LlmModelPrice) {
        self.llm_prices.insert(model, price);
    }

    fn llm_price(&self, model: &str) -> LlmModelPrice {
        self.llm_prices
            .get(model)
            .copied()
            .unwrap_or_else(LlmModelPrice::zero)
    }

    fn llm_account(&mut self, pid: u64, report: LlmUsageReport) -> LlmAccountOutcome {
        // 1) price table lookup
        let price = self.llm_price(&report.model);
        let cost_in = (report.prompt_tokens as u128 * price.prompt_per_1k_micros as u128) / 1_000;
        let cost_out =
            (report.completion_tokens as u128 * price.completion_per_1k_micros as u128) / 1_000;
        // saturate to u64 on overflow (defensive; real usage fits easily)
        let charged_cost_micros: u64 = cost_in
            .saturating_add(cost_out)
            .try_into()
            .unwrap_or(u64::MAX);

        // 2) trace: record the accounting event (best-effort; never fails)
        {
            use crate::types::FastMap;
            let mut fields: FastMap<String, String> = FastMap::default();
            fields.insert("model".to_string(), report.model.clone());
            fields.insert(
                "prompt_tokens".to_string(),
                report.prompt_tokens.to_string(),
            );
            fields.insert(
                "completion_tokens".to_string(),
                report.completion_tokens.to_string(),
            );
            fields.insert(
                "cached_prompt_tokens".to_string(),
                report.cached_prompt_tokens.to_string(),
            );
            fields.insert("latency_ms".to_string(), report.latency_ms.to_string());
            fields.insert("cost_micros".to_string(), charged_cost_micros.to_string());
            <Self as TraceOps>::trace_event(
                self,
                "llm.account".to_string(),
                TraceLevel::Info,
                Some(pid),
                fields,
                None,
            );
        }

        // 3) charge rusage (atomic via rusage_charge, which saturates + enforces limits)
        let delta = ResourceUsageDelta {
            tokens_in: report.prompt_tokens,
            tokens_out: report.completion_tokens,
            cost_micros: charged_cost_micros,
            ..Default::default()
        };
        let verdict = <Self as RlimitOps>::rusage_charge(self, pid, delta);

        // 4) Append one audit record to the bounded ledger (for external drain-and-persist).
        {
            let seq = self.llm_usage.alloc_seq();
            let total_tokens = report
                .prompt_tokens
                .saturating_add(report.completion_tokens);
            self.llm_usage.push(crate::primitives::LlmUsageRecord {
                seq,
                tick: self.tick,
                pid,
                model: report.model,
                prompt_tokens: report.prompt_tokens,
                completion_tokens: report.completion_tokens,
                reasoning_tokens: report.reasoning_tokens.min(report.completion_tokens),
                total_tokens,
                cached_prompt_tokens: report.cached_prompt_tokens,
                latency_ms: report.latency_ms,
                cost_micros: charged_cost_micros,
            });
        }

        LlmAccountOutcome {
            charged_cost_micros,
            verdict,
        }
    }

    fn llm_usage_drain_since(&self, since_seq: u64) -> Vec<crate::primitives::LlmUsageRecord> {
        self.llm_usage.drain_since(since_seq)
    }

    fn llm_usage_head_seq(&self) -> u64 {
        self.llm_usage.head_seq()
    }

    fn llm_usage_set_capacity(&mut self, cap: usize) {
        self.llm_usage.set_capacity(cap);
    }
}
