use super::*;

impl TraceOps for LocalOS {
    fn trace_span_enter(
        &mut self,
        name: String,
        parent: Option<u64>,
        fields: FastMap<String, String>,
    ) -> u64 {
        let span_id = self.trace.alloc_span();
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level: TraceLevel::Debug,
            name,
            span_id: Some(span_id),
            parent_span_id: parent,
            kind: TraceKind::SpanEnter,
            fields: TraceRecord::pack_fields(fields),
            message: None,
        };
        self.trace.push(rec);
        span_id
    }

    fn trace_span_exit(&mut self, span_id: u64, fields: FastMap<String, String>) {
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level: TraceLevel::Debug,
            name: String::new(),
            span_id: Some(span_id),
            parent_span_id: None,
            kind: TraceKind::SpanExit,
            fields: TraceRecord::pack_fields(fields),
            message: None,
        };
        self.trace.push(rec);
    }

    fn trace_event(
        &mut self,
        name: String,
        level: TraceLevel,
        span_id: Option<u64>,
        fields: FastMap<String, String>,
        message: Option<String>,
    ) {
        let seq = self.trace.alloc_seq();
        let rec = TraceRecord {
            seq,
            tick: self.tick,
            pid: self.current_pid,
            level,
            name,
            span_id,
            parent_span_id: None,
            kind: TraceKind::Event,
            fields: TraceRecord::pack_fields(fields),
            message,
        };
        self.trace.push(rec);
    }

    fn trace_recent(&self, n: usize) -> Vec<TraceRecord> {
        self.trace.buf.iter().rev().take(n).cloned().collect()
    }

    fn trace_drain_since(&self, since_seq: u64) -> Vec<TraceRecord> {
        self.trace
            .buf
            .iter()
            .filter(|r| r.seq > since_seq)
            .cloned()
            .collect()
    }

    fn trace_head_seq(&self) -> u64 {
        self.trace.buf.back().map(|r| r.seq).unwrap_or(0)
    }

    fn trace_set_capacity(&mut self, cap: usize) {
        self.trace.capacity = cap;
        while self.trace.buf.len() > cap {
            self.trace.buf.pop_front();
        }
    }
}
