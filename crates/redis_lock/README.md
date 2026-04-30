# Redis Lock - Rust 分布式锁库

类似 Java Redisson 的 Rust 分布式锁实现，基于 Redis 提供可重入锁、公平锁、读写锁等多种分布式锁功能。

## 特性

- **可重入锁（ReentrantLock）**：类似 Redisson 的 RLock，支持同一线程/协程多次获取锁
- **公平锁（FairLock）**：按照请求顺序获取锁（FIFO）
- **读写锁（ReadWriteLock）**：类似 Redisson 的 RReadWriteLock，支持并发读和互斥写
- **Watchdog 自动续期**：防止锁因超时自动释放，后台自动续期
- **灵活的锁选项**：可配置 TTL、获取超时、重试间隔等

## 快速开始

### 1. 添加依赖

在 `Cargo.toml` 中添加：

```toml
[dependencies]
redis_lock = { path = "crates/redis_lock" }
tokio = { version = "1", features = ["full"] }
```

### 2. 使用示例

#### 可重入锁

```rust
use redis_lock::{ReentrantLock, Lock, LockOptions};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 直接通过 Redis 地址创建可重入锁，支持 "127.0.0.1:6381" 或 "redis://127.0.0.1:6381"
    let lock = ReentrantLock::from_url_with_options(
        "my_lock",
        "127.0.0.1:6381",
        LockOptions::new()
            .with_ttl(Duration::from_secs(30))
            .with_watchdog(true),
    ).await?;

    // 获取锁
    lock.lock().await?;
    println!("Lock acquired!");

    // 可重入：同一线程可以再次获取锁
    lock.try_lock().await?;
    println!("Reentrant lock acquired!");

    // 释放锁（需要释放相同的次数）
    lock.unlock().await?;
    lock.unlock().await?;

    Ok(())
}
```

#### 公平锁

```rust
use redis_lock::{FairLock, Lock, LockOptions};
use std::time::Duration;

let fair_lock = FairLock::from_url_with_options(
    "my_fair_lock",
    "127.0.0.1:6381",
    LockOptions::new().with_ttl(Duration::from_secs(10)),
).await?;

// 尝试获取公平锁
if fair_lock.try_lock().await? {
    println!("Fair lock acquired!");
    fair_lock.unlock().await?;
}
```

#### 读写锁

```rust
use redis_lock::{ReadWriteLock, Lock};
use std::time::Duration;

let rwlock = ReadWriteLock::from_url(
    "my_rw_lock",
    "127.0.0.1:6381",
).await?;

// 获取读锁（多个读者可以同时持有）
let read_lock = rwlock.read_lock();
read_lock.lock().await?;
println!("Read lock acquired!");

// 释放读锁
read_lock.unlock().await?;

// 获取写锁（互斥）
let write_lock = rwlock.write_lock();
write_lock.lock().await?;
println!("Write lock acquired!");

// 释放写锁
write_lock.unlock().await?;
```

#### 使用 LockBuilder 简化创建

```rust
use redis_lock::{LockBuilder, ReentrantLock};

let lock: ReentrantLock = LockBuilder::from_url("127.0.0.1:6381").await?
    .ttl(Duration::from_secs(20))
    .acquire_timeout(Duration::from_secs(5))
    .retry_interval(Duration::from_millis(200))
    .watchdog(true)
    .build_reentrant("my_lock");
```

## API 文档

### Lock trait

所有锁都实现 `Lock` trait，提供以下方法：

- `try_lock() -> Result<bool>`：尝试获取锁（非阻塞）
- `lock() -> Result<()>`：获取锁（阻塞直到成功或超时）
- `try_lock_timeout(timeout: Duration) -> Result<bool>`：尝试获取锁，带超时
- `unlock() -> Result<()>`：释放锁
- `is_locked() -> Result<bool>`：检查锁是否被持有
- `remaining_ttl() -> Result<Option<Duration>>`：获取锁的剩余存活时间
- `force_unlock() -> Result<()>`：强制释放锁（不管是否是持有者）

### 直接从 Redis 地址创建锁

- `ReentrantLock::from_url(name, redis_url).await`：使用默认选项创建可重入锁
- `ReentrantLock::from_url_with_options(name, redis_url, options).await`：使用自定义选项创建可重入锁
- `FairLock::from_url(name, redis_url).await`：使用默认选项创建公平锁
- `FairLock::from_url_with_options(name, redis_url, options).await`：使用自定义选项创建公平锁
- `ReadWriteLock::from_url(name, redis_url).await`：使用默认选项创建读写锁
- `ReadWriteLock::from_url_with_options(name, redis_url, options).await`：使用自定义选项创建读写锁
- `LockBuilder::from_url(redis_url).await`：从 Redis 地址创建 builder

`redis_url` 支持完整 URL（如 `redis://127.0.0.1:6381`），也支持裸地址（如 `127.0.0.1:6381`）。

### LockOptions

- `with_ttl(ttl: Duration)`：设置锁的过期时间（默认 30 秒）
- `with_acquire_timeout(timeout: Duration)`：设置获取锁的超时时间（默认 10 秒）
- `with_retry_interval(interval: Duration)`：设置重试间隔（默认 100 毫秒）
- `with_watchdog(enable: bool)`：启用/禁用 Watchdog 自动续期（默认 true）
- `with_watchdog_interval(interval: Duration)`：设置 Watchdog 续期间隔（默认 TTL/3）

## 注意事项

1. **Redis 服务器**：需要运行 Redis 服务器（版本 3.0+）
2. **Lua 脚本**：实现使用 Lua 脚本保证原子性
3. **Watchdog**：自动续期适用于长时间持有的锁，短期锁可以禁用
4. **重入计数**：可重入锁需要释放相同次数，建议使用 RAII 模式
5. **公平锁性能**：公平锁使用队列实现，性能略低于普通锁

## 测试

运行测试（需要本地 Redis 服务器）：

```bash
REDIS_LOCK_TEST_URL=127.0.0.1:6381 cargo test -p redis_lock
```

## 项目结构

```
crates/redis_lock/
├── Cargo.toml
├── src/
│   ├── lib.rs          # 主入口，导出公共 API
│   ├── client.rs       # Redis 客户端封装
│   ├── error.rs        # 错误类型定义
│   ├── lock.rs         # Lock trait 和 LockOptions
│   ├── locks/
│   │   ├── mod.rs
│   │   ├── reentrant.rs  # 可重入锁
│   │   ├── fair.rs      # 公平锁
│   │   └── readwrite.rs # 读写锁
│   └── watchdog.rs     # Watchdog 自动续期
└── tests/
    └── integration_test.rs
```

## 许可证

[MIT License](LICENSE)
