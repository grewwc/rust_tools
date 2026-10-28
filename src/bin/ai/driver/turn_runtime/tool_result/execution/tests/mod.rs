//! Tests for the `execution/` feature clusters, one child module per
//! cluster (`prepare` has no tests of its own).

mod audit_evidence;
mod common;
mod completion_gate;
mod final_citations;
mod final_recovery;
mod final_terminator;
mod followup;
mod iteration;
mod observer;
mod output_format;
mod outstanding;
mod patch_retry;
mod rejection;
mod round;
