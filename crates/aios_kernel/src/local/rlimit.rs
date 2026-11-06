use super::*;

fn candidate_usage(usage: &ResourceUsage, delta: &ResourceUsageDelta) -> ResourceUsage {
    ResourceUsage {
        turns: usage.turns.saturating_add(delta.turns),
        tool_calls: usage.tool_calls.saturating_add(delta.tool_calls),
        tokens_in: usage.tokens_in.saturating_add(delta.tokens_in),
        tokens_out: usage.tokens_out.saturating_add(delta.tokens_out),
        cost_micros: usage.cost_micros.saturating_add(delta.cost_micros),
        last_tool_call_bytes: delta
            .last_tool_call_bytes
            .unwrap_or(usage.last_tool_call_bytes),
        fs_bytes: usage.fs_bytes.saturating_add(delta.fs_bytes),
    }
}

fn usage_verdict(
    usage: &ResourceUsage,
    limits: &ResourceLimit,
    supplied_tool_bytes: Option<u64>,
    elapsed: u64,
) -> RlimitVerdict {
    // Callers act on the first exceeded dimension, so this order is part of the contract.
    for (dimension, used, limit) in [
        (RlimitDim::Turns, Some(usage.turns), limits.max_turns),
        (
            RlimitDim::ToolCalls,
            Some(usage.tool_calls),
            limits.max_tool_calls,
        ),
        (
            RlimitDim::TokensIn,
            Some(usage.tokens_in),
            limits.max_tokens_in,
        ),
        (
            RlimitDim::TokensOut,
            Some(usage.tokens_out),
            limits.max_tokens_out,
        ),
        (
            RlimitDim::CostMicros,
            Some(usage.cost_micros),
            limits.max_cost_micros,
        ),
        // Stored tool bytes are observational; only a supplied size is checked.
        (
            RlimitDim::ToolCallBytes,
            supplied_tool_bytes,
            limits.max_tool_call_bytes,
        ),
        (
            RlimitDim::FsBytes,
            Some(usage.fs_bytes),
            limits.max_fs_bytes,
        ),
        (
            RlimitDim::WallclockTicks,
            Some(elapsed),
            limits.max_wallclock_ticks,
        ),
    ] {
        if let Some(used) = used
            && used > limit
        {
            return RlimitVerdict::Exceeded {
                dimension,
                used,
                limit,
            };
        }
    }
    RlimitVerdict::Ok
}

impl RlimitOps for LocalOS {
    fn rlimit_set(&mut self, pid: u64, limits: ResourceLimit) -> Result<(), String> {
        let proc = self
            .processes
            .get_mut(&pid)
            .ok_or_else(|| format!("rlimit_set: no such pid {}", pid))?;
        if limits.max_turns == u64::MAX {
            proc.quota_turns = 0;
        } else {
            proc.quota_turns = limits.max_turns as usize;
        }
        proc.limits = limits;
        Ok(())
    }

    fn rlimit_get(&self, pid: u64) -> Option<ResourceLimit> {
        self.processes.get(&pid).map(|p| p.limits.clone())
    }

    fn rusage_get(&self, pid: u64) -> Option<ResourceUsage> {
        self.processes.get(&pid).map(|p| p.usage.clone())
    }

    fn rusage_charge(&mut self, pid: u64, delta: ResourceUsageDelta) -> RlimitVerdict {
        let Some(proc) = self.processes.get_mut(&pid) else {
            return RlimitVerdict::NoSuchProcess;
        };
        proc.usage = candidate_usage(&proc.usage, &delta);
        // Keep legacy per-process counters in sync even when the verdict is exceeded.
        proc.turns_used = proc.usage.turns as usize;
        proc.tool_calls_used = proc.usage.tool_calls as usize;

        usage_verdict(
            &proc.usage,
            &proc.limits,
            delta.last_tool_call_bytes,
            self.tick.saturating_sub(proc.created_at_tick),
        )
    }

    fn rlimit_check(&self, pid: u64, delta: &ResourceUsageDelta) -> RlimitVerdict {
        let Some(proc) = self.processes.get(&pid) else {
            return RlimitVerdict::NoSuchProcess;
        };
        let usage = candidate_usage(&proc.usage, delta);
        usage_verdict(
            &usage,
            &proc.limits,
            delta.last_tool_call_bytes,
            self.tick.saturating_sub(proc.created_at_tick),
        )
    }
}
