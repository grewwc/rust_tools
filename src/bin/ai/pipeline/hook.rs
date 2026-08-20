// =============================================================================
// HookRegistry - 流水线钩子注册表（Turn Pipeline Hook）
// =============================================================================
// 为每个 StageKind 提供 before / after 钩子插槽，解耦 driver 中散落的
// 钩子调用（如 hooks::run_pre_turn / post_turn）。钩子为纯内存回调，
// 不依赖子进程，适合单测与中间件注入。

use std::collections::HashMap;
use super::context::{PipelineContext, StageKind};

/// 钩子回调签名：同步、可失败；需要异步的用 block_on / 中间件链。
pub type HookFn = Box<dyn Fn(&mut PipelineContext<'_>) -> Result<(), Box<dyn std::error::Error + Send + Sync>> + Send + Sync>;

/// 单个已注册钩子的元数据（便于调试与去重）
pub struct HookEntry {
    pub name: &'static str,
    pub func: HookFn,
}

/// 按 StageKind + 时机（before/after）组织的注册表
#[derive(Default)]
pub struct HookRegistry {
    before: HashMap<StageKind, Vec<HookEntry>>,
    after: HashMap<StageKind, Vec<HookEntry>>,
    /// 全局 turn 级钩子（不绑定具体 stage），如 on_turn_start / on_turn_end
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

    /// 触发指定 stage 的 before 钩子（按注册顺序），遇错短路返回 Err
    pub fn fire_before(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // 让 ctx.stage 与正在触发的 stage 保持一致：无论调用方（runner / driver 钩子）
        // 是否手动 advance，钩子观察到的 stage 都是真实的触发点，避免读到 new() 的 Prepare 默认值。
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

    /// 仅触发某个 stage 自身的 before 钩子（不含全局钩子，避免与 turn 级全局钩子重复触发）。
    pub fn fire_stage_before(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ctx.stage = kind;
        for h in self.before.get(&kind).into_iter().flatten() {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    /// 仅触发某个 stage 自身的 after 钩子（不含全局钩子）。
    pub fn fire_stage_after(&self, ctx: &mut PipelineContext<'_>, kind: StageKind) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ctx.stage = kind;
        for h in self.after.get(&kind).into_iter().flatten() {
            (h.func)(ctx)?;
        }
        Ok(())
    }

    /// 兼容旧 shell hooks 的空注册表快捷构造
    pub fn is_empty(&self) -> bool {
        self.before.is_empty() && self.after.is_empty() && self.global_before.is_empty() && self.global_after.is_empty()
    }

    pub fn len_before(&self, kind: StageKind) -> usize {
        self.before.get(&kind).map(|v| v.len()).unwrap_or(0)
    }
    pub fn len_after(&self, kind: StageKind) -> usize {
        self.after.get(&kind).map(|v| v.len()).unwrap_or(0)
    }

    /// 全局钩子触发（不绑定 StageKind）
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
        // 复用 middleware 的合法 App 构造，避免 zeroed 对 PathBuf/Arc 等非零类型的 UB。
        // pipeline 测试仅读写 tags，不依赖 App 字段内容。
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
        // 泄漏的 App 不回收，避免 zeroed Drop 的 UB
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
