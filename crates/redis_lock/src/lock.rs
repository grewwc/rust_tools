use async_trait::async_trait;
use std::time::Duration;

/// 锁的选项配置
#[derive(Debug, Clone)]
pub struct LockOptions {
    /// 锁的过期时间（TTL），默认 30 秒
    pub ttl: Duration,
    /// 获取锁的超时时间，默认 10 秒
    pub acquire_timeout: Duration,
    /// 重试间隔，默认 100 毫秒
    pub retry_interval: Duration,
    /// 是否启用自动续期（Watchdog），默认 true
    pub enable_watchdog: bool,
    /// Watchdog 续期间隔，默认锁 TTL 的 1/3
    pub watchdog_interval: Option<Duration>,
}

impl Default for LockOptions {
    fn default() -> Self {
        Self {
            ttl: Duration::from_secs(30),
            acquire_timeout: Duration::from_secs(10),
            retry_interval: Duration::from_millis(100),
            enable_watchdog: true,
            watchdog_interval: None,
        }
    }
}

impl LockOptions {
    /// 创建默认配置
    pub fn new() -> Self {
        Self::default()
    }

    /// 设置锁的过期时间
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// 设置获取锁的超时时间
    pub fn with_acquire_timeout(mut self, timeout: Duration) -> Self {
        self.acquire_timeout = timeout;
        self
    }

    /// 设置重试间隔
    pub fn with_retry_interval(mut self, interval: Duration) -> Self {
        self.retry_interval = interval;
        self
    }

    /// 启用或禁用自动续期
    pub fn with_watchdog(mut self, enable: bool) -> Self {
        self.enable_watchdog = enable;
        self
    }

    /// 设置 Watchdog 续期间隔
    pub fn with_watchdog_interval(mut self, interval: Duration) -> Self {
        self.watchdog_interval = Some(interval);
        self
    }
}

/// 分布式锁的核心 trait，类似 Redisson 的 RLock
/// 使用 async_trait 支持异步操作
#[async_trait]
pub trait Lock: Send + Sync {
    /// 尝试获取锁（非阻塞）
    async fn try_lock(&self) -> crate::error::Result<bool>;

    /// 获取锁（阻塞，直到获取成功或超时）
    async fn lock(&self) -> crate::error::Result<()>;

    /// 尝试获取锁，带超时
    async fn try_lock_timeout(&self, timeout: Duration) -> crate::error::Result<bool>;

    /// 释放锁
    async fn unlock(&self) -> crate::error::Result<()>;

    /// 检查锁是否被持有
    async fn is_locked(&self) -> crate::error::Result<bool>;

    /// 获取锁的剩余存活时间
    async fn remaining_ttl(&self) -> crate::error::Result<Option<Duration>>;

    /// 强制释放锁（不管是否是锁的持有者）
    async fn force_unlock(&self) -> crate::error::Result<()>;
}
