// =============================================================================
// HookRegistry - pipeline hook registry (Turn Pipeline Hook)
// =============================================================================
// Provides before/after hook slots for each StageKind, decoupling the hook
// calls scattered through the driver (such as hooks::run_pre_turn /
// post_turn). Hooks are pure in-memory callbacks with no subprocess
// dependency, suitable for unit tests and middleware injection.

use std::collections::HashMap;
use super::context::{PipelineContext, StageKind};

/// Hook callback signature: synchronous and fallible; for async, use
/// block_on / the middleware chain.
pub type HookFn = Box<dyn Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync>;

/// Metadata for a single registered hook (for debugging and deduplication).
pub struct HookEntry {
    pub name: &'static str,
    pub func: HookFn,
}

/// Registry organized by StageKind + timing (before/after).
#[derive(Default)]
pub struct HookRegistry {
    before: HashMap<StageKind, Vec<HookEntry>>,
    after: HashMap<StageKind, Vec<HookEntry>>,
    /// Global turn-level hooks (not bound to a specific stage), such as
    /// on_turn_start / on_turn_end.
    global_before: Vec<HookEntry>,
    global_after: Vec<HookEntry>,
}

impl HookRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register_before<F>(&mut self, kind: StageKind, name: &'static str, f: F)
    where
        F: Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
    {
        self.before.entry(kind).or_default().push(HookEntry { name, func: Box::new(f) });
    }

    pub fn register_after<F>(&mut self, kind: StageKind, name: &'static str, f: F)
    where
        F: Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
    {
        self.after.entry(kind).or_default().push(HookEntry { name, func: Box::new(f) });
    }

    pub fn register_global_before<F>(&mut self, name: &'static str, f: F)
    where
        F: Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
    {
        self.global_before.push(HookEntry { name, func: Box::new(f) });
    }

    pub fn register_global_after<F>(&mut self, name: &'static str, f: F)
    where
        F: Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync + 'static,
    {
        self.global_after.push(HookEntry { name, func: Box::new(f) });
    }

    /// Fires the before hooks for the given stage (in registration order),
    /// short-circuiting with Err on the first failure.
    pub fn fire_before(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Keep ctx.stage aligned with the stage being fired: regardless of
        // whether the caller (runner / driver hooks) advances manually, hooks
        // observe the real trigger point instead of the Prepare default from
        // new().
        ctx.stage = kind;
        for h in self.global_before.iter().chain(self.before.get(&kind).into_iter().flatten()) {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    pub fn fire_after(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ctx.stage = kind;
        for h in self.after.get(&kind).into_iter().flatten().chain(self.global_after.iter()) {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    /// Fires only a stage's own before hooks (excluding global hooks, to avoid
    /// double-firing turn-level global hooks).
    pub fn fire_stage_before(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ctx.stage = kind;
        for h in self.before.get(&kind).into_iter().flatten() {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    /// Fires only a stage's own after hooks (excluding global hooks).
    pub fn fire_stage_after(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ctx.stage = kind;
        for h in self.after.get(&kind).into_iter().flatten() {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    /// Quick construction of an empty registry compatible with legacy shell hooks.
    pub fn is_empty(&self) -> bool {
        self.before.is_empty() && self.after.is_empty() && self.global_before.is_empty() && self.global_after.is_empty()
    }

    pub fn len_before(&self, kind: StageKind) -> usize {
        self.before.get(&kind).map(|v| v.len()).unwrap_or(0)
    }
    pub fn len_after(&self, kind: StageKind) -> usize {
        self.after.get(&kind).map(|v| v.len()).unwrap_or(0)
    }

    /// Fires global hooks (not bound to a StageKind).
    pub fn fire_global_before(&self, ctx: &mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for h in &self.global_before { (h.func)(ctx)?; }
        Ok(())
    }
    pub fn fire_global_after(&self, ctx: &mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        for h in &self.global_after { (h.func)(ctx)?; }
        Ok(())
    }
    pub fn len(&self) -> usize {
        self.before.values().map(|v| v.len()).sum::<usize>()
            + self.after.values().map(|v| v.len()).sum::<usize>()
            + self.global_before.len()
            + self.global_after.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::App;
    fn leak_app() -> &'static mut App {
        // Reuse middleware's legitimate App construction to avoid the UB of
        // zeroing non-zero-sized types such as PathBuf/Arc.
        // The pipeline tests only read/write tags and do not rely on App
        // field contents.
        let app = crate::ai::middleware::test_util::test_app();
        Box::leak(Box::new(app))
    }

    #[test]
    fn hook_fires_in_order() {
        let mut reg = HookRegistry::new();
        reg.register_before(StageKind::Prepare, "a", |ctx| { ctx.tags.push("a".into()); Ok(()) });
        reg.register_before(StageKind::Prepare, "b", |ctx| { ctx.tags.push("b".into()); Ok(()) });
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        reg.fire_before(&mut ctx, StageKind::Prepare).unwrap();
        assert_eq!(ctx.tags, vec!["a", "b"]);
        // The leaked App is never reclaimed, avoiding zeroed Drop UB.
    }

    #[test]
    fn hook_short_circuits_on_error() {
        let mut reg = HookRegistry::new();
        reg.register_before(StageKind::Prepare, "fail", |_| Err(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "boom")) as _));
        reg.register_before(StageKind::Prepare, "never", |ctx| { ctx.tags.push("never".into()); Ok(()) });
        let app = leak_app();
        let mut ctx = PipelineContext::new(app, vec![], 0);
        assert!(reg.fire_before(&mut ctx, StageKind::Prepare).is_err());
        assert!(ctx.tags.is_empty());
    }
}
