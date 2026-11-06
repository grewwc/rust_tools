use super::*;

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
        proc.usage.turns = proc.usage.turns.saturating_add(delta.turns);
        proc.usage.tool_calls = proc.usage.tool_calls.saturating_add(delta.tool_calls);
        proc.usage.tokens_in = proc.usage.tokens_in.saturating_add(delta.tokens_in);
        proc.usage.tokens_out = proc.usage.tokens_out.saturating_add(delta.tokens_out);
        proc.usage.cost_micros = proc.usage.cost_micros.saturating_add(delta.cost_micros);
        proc.usage.fs_bytes = proc.usage.fs_bytes.saturating_add(delta.fs_bytes);
        if let Some(b) = delta.last_tool_call_bytes {
            proc.usage.last_tool_call_bytes = b;
        }
        // keep legacy per-process counters in sync
        proc.turns_used = proc.usage.turns as usize;
        proc.tool_calls_used = proc.usage.tool_calls as usize;

        let lim = &proc.limits;
        if proc.usage.turns > lim.max_turns {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::Turns,
                used: proc.usage.turns,
                limit: lim.max_turns,
            };
        }
        if proc.usage.tool_calls > lim.max_tool_calls {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::ToolCalls,
                used: proc.usage.tool_calls,
                limit: lim.max_tool_calls,
            };
        }
        if proc.usage.tokens_in > lim.max_tokens_in {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensIn,
                used: proc.usage.tokens_in,
                limit: lim.max_tokens_in,
            };
        }
        if proc.usage.tokens_out > lim.max_tokens_out {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensOut,
                used: proc.usage.tokens_out,
                limit: lim.max_tokens_out,
            };
        }
        if proc.usage.cost_micros > lim.max_cost_micros {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::CostMicros,
                used: proc.usage.cost_micros,
                limit: lim.max_cost_micros,
            };
        }
        if let Some(b) = delta.last_tool_call_bytes {
            if b > lim.max_tool_call_bytes {
                return RlimitVerdict::Exceeded {
                    dimension: RlimitDim::ToolCallBytes,
                    used: b,
                    limit: lim.max_tool_call_bytes,
                };
            }
        }
        if proc.usage.fs_bytes > lim.max_fs_bytes {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::FsBytes,
                used: proc.usage.fs_bytes,
                limit: lim.max_fs_bytes,
            };
        }
        // wallclock: elapsed = self.tick - created_at_tick
        let elapsed = self.tick.saturating_sub(proc.created_at_tick);
        if elapsed > lim.max_wallclock_ticks {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::WallclockTicks,
                used: elapsed,
                limit: lim.max_wallclock_ticks,
            };
        }
        RlimitVerdict::Ok
    }

    fn rlimit_check(&self, pid: u64, delta: &ResourceUsageDelta) -> RlimitVerdict {
        let Some(proc) = self.processes.get(&pid) else {
            return RlimitVerdict::NoSuchProcess;
        };
        let lim = &proc.limits;
        let new_turns = proc.usage.turns.saturating_add(delta.turns);
        if new_turns > lim.max_turns {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::Turns,
                used: new_turns,
                limit: lim.max_turns,
            };
        }
        let new_calls = proc.usage.tool_calls.saturating_add(delta.tool_calls);
        if new_calls > lim.max_tool_calls {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::ToolCalls,
                used: new_calls,
                limit: lim.max_tool_calls,
            };
        }
        let new_in = proc.usage.tokens_in.saturating_add(delta.tokens_in);
        if new_in > lim.max_tokens_in {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensIn,
                used: new_in,
                limit: lim.max_tokens_in,
            };
        }
        let new_out = proc.usage.tokens_out.saturating_add(delta.tokens_out);
        if new_out > lim.max_tokens_out {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::TokensOut,
                used: new_out,
                limit: lim.max_tokens_out,
            };
        }
        let new_cost = proc.usage.cost_micros.saturating_add(delta.cost_micros);
        if new_cost > lim.max_cost_micros {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::CostMicros,
                used: new_cost,
                limit: lim.max_cost_micros,
            };
        }
        if let Some(b) = delta.last_tool_call_bytes {
            if b > lim.max_tool_call_bytes {
                return RlimitVerdict::Exceeded {
                    dimension: RlimitDim::ToolCallBytes,
                    used: b,
                    limit: lim.max_tool_call_bytes,
                };
            }
        }
        let new_fs = proc.usage.fs_bytes.saturating_add(delta.fs_bytes);
        if new_fs > lim.max_fs_bytes {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::FsBytes,
                used: new_fs,
                limit: lim.max_fs_bytes,
            };
        }
        let elapsed = self.tick.saturating_sub(proc.created_at_tick);
        if elapsed > lim.max_wallclock_ticks {
            return RlimitVerdict::Exceeded {
                dimension: RlimitDim::WallclockTicks,
                used: elapsed,
                limit: lim.max_wallclock_ticks,
            };
        }
        RlimitVerdict::Ok
    }
}
