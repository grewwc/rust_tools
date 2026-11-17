//! Turn-local model-derived premise bookkeeping with runtime-bound evidence.
//!
//! The synchronous executor only validates arguments. The driver must apply the
//! provisional update against its current canonical evidence catalogue and replace
//! the tool result with the receipt before appending it to canonical history.

#[path = "registry/critical_premises.rs"]
mod registration;
#[path = "service/critical_premises.rs"]
mod service;
#[path = "storage/critical_premises.rs"]
mod state;

#[cfg(test)]
pub(in crate::ai) use service::execute;
pub(in crate::ai) use service::validated_from_provisional;
pub(in crate::ai) use state::{
    Coverage, EvidenceCatalog, EvidenceRecord, ObservedObject, PremiseState,
};
