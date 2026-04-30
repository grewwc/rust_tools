//! Redis 分布式锁库，类似 Java Redisson 的功能
//!
//! 提供可重入锁、公平锁、读写锁等多种分布式锁实现
//! 支持自动续期（Watchdog 机制）和灵活的锁选项
//!
//! # 使用示例
//! ```rust,no_run
//! use redis_lock::{ReentrantLock, Lock, LockOptions};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // 创建可重入锁
//!     let lock = ReentrantLock::from_url_with_options(
//!         "my_lock",
//!         "127.0.0.1:6381",
//!         LockOptions::new()
//!             .with_ttl(Duration::from_secs(30))
//!             .with_watchdog(true)
//!     ).await?;
//!
//!     // 获取锁
//!     lock.lock().await?;
//!     
//!     // 执行业务逻辑
//!     println!("Lock acquired!");
//!     
//!     // 释放锁
//!     lock.unlock().await?;
//!     
//!     Ok(())
//! }
//! ```

use std::sync::Arc;
use std::time::Duration;

pub mod client;
pub mod error;
pub mod lock;
pub mod locks;
pub mod watchdog;

pub use client::RedisLockClient as RedisClient;
pub use error::{Error, Result};
pub use lock::{Lock, LockOptions};
pub use locks::{FairLock, ReadLock, ReadWriteLock, ReentrantLock, WriteLock};
pub use watchdog::Watchdog;

/// 便捷的锁构建器
pub struct LockBuilder {
    client: Arc<RedisClient>,
    options: LockOptions,
}

impl LockBuilder {
    /// 创建新的锁构建器
    pub fn new(client: Arc<RedisClient>) -> Self {
        Self {
            client,
            options: LockOptions::default(),
        }
    }

    /// 通过 Redis URL 创建锁构建器。
    pub async fn from_url(redis_url: &str) -> Result<Self> {
        let client = Arc::new(RedisClient::from_url(redis_url).await?);
        Ok(Self::new(client))
    }

    /// 设置锁的过期时间
    pub fn ttl(mut self, ttl: Duration) -> Self {
        self.options.ttl = ttl;
        self
    }

    /// 设置获取锁的超时时间
    pub fn acquire_timeout(mut self, timeout: Duration) -> Self {
        self.options.acquire_timeout = timeout;
        self
    }

    /// 设置重试间隔
    pub fn retry_interval(mut self, interval: Duration) -> Self {
        self.options.retry_interval = interval;
        self
    }

    /// 启用或禁用 Watchdog
    pub fn watchdog(mut self, enable: bool) -> Self {
        self.options.enable_watchdog = enable;
        self
    }

    /// 构建可重入锁
    pub fn build_reentrant(self, name: &str) -> ReentrantLock {
        ReentrantLock::new(name, self.client, self.options)
    }

    /// 构建公平锁
    pub fn build_fair(self, name: &str) -> FairLock {
        FairLock::new(name, self.client, self.options)
    }

    /// 构建读写锁
    pub fn build_readwrite(self, name: &str) -> ReadWriteLock {
        ReadWriteLock::new(name, self.client, self.options)
    }
}
