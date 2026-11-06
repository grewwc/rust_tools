use super::*;

// ------------------------------------------------------------------
// Phase 0 primitives: futex + trace
// ------------------------------------------------------------------

#[test]
fn futex_basic_create_load_store_cas() {
    use crate::primitives::FutexOps;
    let mut os = LocalOS::new();
    let addr = os.futex_create(0, "stream_cancel".to_string());
    assert_eq!(os.futex_load(addr), Some(0));
    assert_eq!(os.futex_store(addr, 1), Some(0));
    assert_eq!(os.futex_load(addr), Some(1));
    assert!(os.futex_cas(addr, 1, 2).is_ok());
    assert!(os.futex_cas(addr, 1, 3).is_err());
    assert_eq!(os.futex_load(addr), Some(2));
}

#[test]
fn futex_wake_moves_waiter_to_ready() {
    use crate::primitives::FutexOps;
    let mut os = LocalOS::new();
    let root = os.begin_foreground("fg".to_string(), "goal".to_string(), 10, usize::MAX, None);
    let worker = os
        .spawn(
            Some(root),
            "w".to_string(),
            "do".to_string(),
            20,
            4,
            None,
            None,
        )
        .unwrap();
    // Simulate worker going to sleep on a futex
    if let Some(p) = os.processes.get_mut(&worker) {
        p.state = ProcessState::Waiting {
            reason: WaitReason::ProcessExit { on_pid: root },
        };
    }
    os.ready_set.remove(&worker);

    let addr = os.futex_create(0, "ready_bell".to_string());
    let seq_before = os.futex_register_waiter(addr, worker).unwrap();
    let woken = os.futex_wake(addr, 1);
    assert_eq!(woken, 1);
    let seq_after = os.futex_seq(addr).unwrap();
    assert!(seq_after > seq_before);
    assert_eq!(
        os.processes.get(&worker).unwrap().state,
        ProcessState::Ready
    );
    assert!(os.ready_set.contains(&worker));
}

#[test]
fn futex_try_wait_reports_value_changed() {
    use crate::primitives::{FutexOps, FutexWakeReason};
    let mut os = LocalOS::new();
    let addr = os.futex_create(0, "t".to_string());
    assert!(
        os.futex_try_wait(addr, 0).is_none(),
        "should block when equal"
    );
    os.futex_store(addr, 7);
    assert_eq!(
        os.futex_try_wait(addr, 0),
        Some(FutexWakeReason::ValueChanged)
    );
}

#[test]
fn trace_records_spans_and_events_in_order() {
    use crate::primitives::{TraceKind, TraceLevel, TraceOps};
    use crate::types::FastMap;
    let mut os = LocalOS::new();
    let _fg = os.begin_foreground("fg".to_string(), "g".to_string(), 10, usize::MAX, None);

    let span = os.trace_span_enter("turn.run".to_string(), None, FastMap::default());
    let mut fields: FastMap<String, String> = FastMap::default();
    fields.insert("model".to_string(), "gpt".to_string());
    os.trace_event(
        "llm.submit".to_string(),
        TraceLevel::Info,
        Some(span),
        fields,
        Some("sent".to_string()),
    );
    os.trace_span_exit(span, FastMap::default());

    let recs = os.trace_drain_since(0);
    assert_eq!(recs.len(), 3);
    assert!(matches!(recs[0].kind, TraceKind::SpanEnter));
    assert!(matches!(recs[1].kind, TraceKind::Event));
    assert!(matches!(recs[2].kind, TraceKind::SpanExit));
    assert_eq!(recs[1].name, "llm.submit");
    assert_eq!(
        recs[1]
            .fields()
            .and_then(|f| f.get("model"))
            .map(String::as_str),
        Some("gpt")
    );
}

#[test]
fn trace_ring_respects_capacity() {
    use crate::primitives::{TraceLevel, TraceOps};
    use crate::types::FastMap;
    let mut os = LocalOS::new();
    os.trace_set_capacity(4);
    for i in 0..10 {
        os.trace_event(
            format!("evt.{}", i),
            TraceLevel::Debug,
            None,
            FastMap::default(),
            None,
        );
    }
    let recs = os.trace_drain_since(0);
    assert_eq!(recs.len(), 4);
    // oldest kept should be evt.6 (after dropping 0..=5)
    assert_eq!(recs[0].name, "evt.6");
    assert_eq!(recs[3].name, "evt.9");
}

#[test]
fn rlimit_set_and_get_roundtrips() {
    use crate::primitives::{ResourceLimit, RlimitOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let mut lim = ResourceLimit::unlimited();
    lim.max_turns = 7;
    lim.max_tool_calls = 3;
    lim.max_tokens_in = 1000;
    os.rlimit_set(pid, lim.clone()).unwrap();
    let got = os.rlimit_get(pid).unwrap();
    assert_eq!(got, lim);
    // quota_turns mirror must be synced too
    assert_eq!(os.get_process(pid).unwrap().quota_turns, 7);
}

#[test]
fn rusage_charge_enforces_turns_limit() {
    use crate::primitives::{
        ResourceLimit, ResourceUsageDelta, RlimitDim, RlimitOps, RlimitVerdict,
    };
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let mut lim = ResourceLimit::unlimited();
    lim.max_turns = 2;
    os.rlimit_set(pid, lim).unwrap();

    assert_eq!(
        os.rusage_charge(
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            }
        ),
        RlimitVerdict::Ok
    );
    assert_eq!(
        os.rusage_charge(
            pid,
            ResourceUsageDelta {
                turns: 1,
                ..Default::default()
            }
        ),
        RlimitVerdict::Ok
    );
    match os.rusage_charge(
        pid,
        ResourceUsageDelta {
            turns: 1,
            ..Default::default()
        },
    ) {
        RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        } => {
            assert_eq!(dimension, RlimitDim::Turns);
            assert_eq!(used, 3);
            assert_eq!(limit, 2);
        }
        v => panic!("expected Exceeded Turns, got {:?}", v),
    }
    // legacy mirror stays in sync
    assert_eq!(os.get_process(pid).unwrap().turns_used, 3);
    assert_eq!(os.rusage_get(pid).unwrap().turns, 3);
}

#[test]
fn rlimit_check_is_pure() {
    use crate::primitives::{
        ResourceLimit, ResourceUsageDelta, RlimitDim, RlimitOps, RlimitVerdict,
    };
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    let mut lim = ResourceLimit::unlimited();
    lim.max_tokens_in = 100;
    os.rlimit_set(pid, lim).unwrap();
    // pre-check a big prompt
    let probe = ResourceUsageDelta {
        tokens_in: 200,
        ..Default::default()
    };
    match os.rlimit_check(pid, &probe) {
        RlimitVerdict::Exceeded { dimension, .. } => {
            assert_eq!(dimension, RlimitDim::TokensIn);
        }
        v => panic!("expected Exceeded TokensIn, got {:?}", v),
    }
    // usage must NOT have moved
    assert_eq!(os.rusage_get(pid).unwrap().tokens_in, 0);
}

#[test]
fn increment_helpers_route_through_rusage_charge() {
    use crate::primitives::RlimitOps;
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.increment_turns_used_for(pid);
    os.increment_turns_used_for(pid);
    os.increment_tool_calls_used_for(pid);
    let u = os.rusage_get(pid).unwrap();
    assert_eq!(u.turns, 2);
    assert_eq!(u.tool_calls, 1);
    // legacy mirrors stay in sync
    let p = os.get_process(pid).unwrap();
    assert_eq!(p.turns_used, 2);
    assert_eq!(p.tool_calls_used, 1);
}

#[test]
fn llm_account_charges_cost_and_updates_rusage() {
    use crate::primitives::{LlmModelPrice, LlmOps, LlmUsageReport, RlimitOps, RlimitVerdict};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    // 1000 prompt tok => 2500 micros; 500 completion tok => 3000 micros.
    os.llm_set_price(
        "gpt-test".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 2_500,
            completion_per_1k_micros: 6_000,
        },
    );
    let out = os.llm_account(
        pid,
        LlmUsageReport {
            model: "gpt-test".into(),
            prompt_tokens: 1_000,
            completion_tokens: 500,
            reasoning_tokens: 0,
            cached_prompt_tokens: 100,
            latency_ms: 42,
        },
    );
    assert_eq!(out.charged_cost_micros, 2_500 + 3_000);
    assert_eq!(out.verdict, RlimitVerdict::Ok);
    let u = os.rusage_get(pid).unwrap();
    assert_eq!(u.tokens_in, 1_000);
    assert_eq!(u.tokens_out, 500);
    assert_eq!(u.cost_micros, 5_500);
}

#[test]
fn llm_account_with_unknown_model_is_free_but_still_charges_tokens() {
    use crate::primitives::RlimitOps;
    use crate::primitives::{LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    // No price registered for "mystery-model"
    let out = os.llm_account(
        pid,
        LlmUsageReport {
            model: "mystery-model".into(),
            prompt_tokens: 123,
            completion_tokens: 45,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 0,
        },
    );
    assert_eq!(out.charged_cost_micros, 0);
    let u = os.rusage_get(pid).unwrap();
    assert_eq!(u.tokens_in, 123);
    assert_eq!(u.tokens_out, 45);
    assert_eq!(u.cost_micros, 0);
}

#[test]
fn llm_account_respects_cost_rlimit() {
    use crate::primitives::{
        LlmModelPrice, LlmOps, LlmUsageReport, ResourceLimit, RlimitDim, RlimitOps, RlimitVerdict,
    };
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_set_price(
        "g".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 1_000,
            completion_per_1k_micros: 0,
        },
    );
    // cost budget = 500 micros. 1000 prompt tokens -> 1000 micros -> Exceeded.
    let mut lim = ResourceLimit::unlimited();
    lim.max_cost_micros = 500;
    os.rlimit_set(pid, lim).unwrap();
    let out = os.llm_account(
        pid,
        LlmUsageReport {
            model: "g".into(),
            prompt_tokens: 1_000,
            completion_tokens: 0,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 0,
        },
    );
    match out.verdict {
        RlimitVerdict::Exceeded {
            dimension,
            used,
            limit,
        } => {
            assert_eq!(dimension, RlimitDim::CostMicros);
            assert_eq!(used, 1_000);
            assert_eq!(limit, 500);
        }
        v => panic!("expected Exceeded CostMicros, got {:?}", v),
    }
}

#[test]
fn llm_account_emits_trace_event() {
    use crate::primitives::{LlmOps, LlmUsageReport, TraceOps};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_account(
        pid,
        LlmUsageReport {
            model: "m".into(),
            prompt_tokens: 10,
            completion_tokens: 5,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 77,
        },
    );
    let recs = os.trace_drain_since(0);
    let found = recs.iter().any(|r| r.name == "llm.account");
    assert!(found, "expected a trace event named llm.account");
}

#[test]
fn llm_usage_ledger_records_and_drains() {
    use crate::primitives::{LlmModelPrice, LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_set_price(
        "m".into(),
        LlmModelPrice {
            prompt_per_1k_micros: 1_000,
            completion_per_1k_micros: 2_000,
        },
    );
    // The initial ledger is empty.
    assert_eq!(os.llm_usage_head_seq(), 0);
    assert!(os.llm_usage_drain_since(0).is_empty());

    os.llm_account(
        pid,
        LlmUsageReport {
            model: "m".into(),
            prompt_tokens: 100,
            completion_tokens: 50,
            reasoning_tokens: 30,
            cached_prompt_tokens: 10,
            latency_ms: 7,
        },
    );
    os.llm_account(
        pid,
        LlmUsageReport {
            model: "m".into(),
            prompt_tokens: 200,
            completion_tokens: 80,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            latency_ms: 0,
        },
    );

    // Full drain: two records, ascending, fields correct.
    let all = os.llm_usage_drain_since(0);
    assert_eq!(all.len(), 2);
    assert_eq!(all[0].seq, 1);
    assert_eq!(all[0].pid, pid);
    assert_eq!(all[0].model, "m");
    assert_eq!(all[0].prompt_tokens, 100);
    assert_eq!(all[0].completion_tokens, 50);
    assert_eq!(all[0].reasoning_tokens, 30);
    assert_eq!(all[0].total_tokens, 150);
    assert_eq!(all[0].cached_prompt_tokens, 10);
    assert_eq!(all[0].latency_ms, 7);
    // 100 prompt -> 100 micros; 50 completion -> 100 micros.
    assert_eq!(all[0].cost_micros, 200);
    assert_eq!(all[1].seq, 2);
    assert_eq!(all[1].total_tokens, 280);

    // Cursor drain: fetch only records with seq>1.
    assert_eq!(os.llm_usage_head_seq(), 2);
    let tail = os.llm_usage_drain_since(1);
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].seq, 2);
    // Draining does not consume; repeated drains return the same result.
    assert_eq!(os.llm_usage_drain_since(0).len(), 2);
}

#[test]
fn llm_usage_ledger_capacity_evicts_oldest() {
    use crate::primitives::{LlmOps, LlmUsageReport};
    let mut os = LocalOS::new();
    let pid = os.begin_foreground("p".into(), "g".into(), 10, 0, None);
    os.llm_usage_set_capacity(2);
    for i in 0..5u64 {
        os.llm_account(
            pid,
            LlmUsageReport {
                model: "m".into(),
                prompt_tokens: i,
                completion_tokens: 0,
                reasoning_tokens: 0,
                cached_prompt_tokens: 0,
                latency_ms: 0,
            },
        );
    }
    // seq still monotonically reaches 5, but only the last two records are kept.
    assert_eq!(os.llm_usage_head_seq(), 5);
    let recs = os.llm_usage_drain_since(0);
    assert_eq!(recs.len(), 2);
    assert_eq!(recs[0].seq, 4);
    assert_eq!(recs[1].seq, 5);
}
