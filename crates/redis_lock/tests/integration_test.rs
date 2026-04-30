//! 集成测试 - 需要本地 Redis 服务器运行
//! 运行测试: cargo test -p redis_lock

use redis_lock::{
    FairLock, Lock, LockBuilder, LockOptions, ReadWriteLock, RedisClient, ReentrantLock,
};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn test_redis_url() -> String {
    std::env::var("REDIS_LOCK_TEST_URL").unwrap_or_else(|_| "127.0.0.1:6381".to_string())
}

/// 测试辅助函数：创建 Redis 客户端
async fn create_test_client() -> Option<Arc<RedisClient>> {
    // 使用用户提供的 Redis 地址
    let redis_url = test_redis_url();
    match tokio::time::timeout(Duration::from_secs(1), RedisClient::from_url(&redis_url)).await {
        Ok(Ok(client)) => Some(Arc::new(client)),
        Ok(Err(_)) | Err(_) => {
            println!(
                "Warning: Redis server not available at {}, skipping test",
                redis_url
            );
            None
        }
    }
}

#[tokio::test]
async fn test_reentrant_lock_try_lock() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock = ReentrantLock::new(
        "test_reentrant_lock",
        client.clone(),
        LockOptions::new().with_ttl(Duration::from_secs(5)),
    );

    // 清理可能存在的旧锁
    let _ = lock.force_unlock().await;

    // 尝试获取锁
    let result = lock.try_lock().await.unwrap();
    assert!(result, "Should acquire lock successfully");

    // 同一线程应该可以重入
    let result2 = lock.try_lock().await.unwrap();
    assert!(result2, "Should reentrant acquire lock");

    // 释放一次
    lock.unlock().await.unwrap();

    // 锁应该还在（因为重入了两次）
    assert!(lock.is_locked().await.unwrap());

    // 再次释放
    lock.unlock().await.unwrap();

    // 现在锁应该被释放了
    assert!(!lock.is_locked().await.unwrap());

    // 清理
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_direct_url_constructors_accept_bare_redis_address() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_watchdog(false);

    let reentrant = match ReentrantLock::from_url_with_options(
        "test_direct_url_reentrant",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let fair = FairLock::from_url_with_options("test_direct_url_fair", &redis_url, options.clone())
        .await
        .unwrap();
    let rwlock =
        ReadWriteLock::from_url_with_options("test_direct_url_rw", &redis_url, options.clone())
            .await
            .unwrap();

    let _ = reentrant.force_unlock().await;
    let _ = fair.force_unlock().await;
    let _ = rwlock.read_lock().force_unlock().await;
    let _ = rwlock.write_lock().force_unlock().await;

    assert!(reentrant.try_lock().await.unwrap());
    reentrant.unlock().await.unwrap();

    assert!(fair.try_lock().await.unwrap());
    fair.unlock().await.unwrap();

    assert!(rwlock.read_lock().try_lock().await.unwrap());
    rwlock.read_lock().unlock().await.unwrap();

    let _ = reentrant.force_unlock().await;
    let _ = fair.force_unlock().await;
    let _ = rwlock.read_lock().force_unlock().await;
    let _ = rwlock.write_lock().force_unlock().await;
}

#[tokio::test]
async fn test_lock_builder_from_url_creates_reentrant_lock() {
    let redis_url = test_redis_url();
    let builder = match LockBuilder::from_url(&redis_url).await {
        Ok(builder) => builder,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };

    let lock = builder
        .ttl(Duration::from_secs(5))
        .watchdog(false)
        .build_reentrant("test_builder_from_url_lock");

    let _ = lock.force_unlock().await;
    assert!(lock.try_lock().await.unwrap());
    lock.unlock().await.unwrap();
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_reentrant_lock_contention_between_distinct_url_locks() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let lock_a = match ReentrantLock::from_url_with_options(
        "test_reentrant_url_contention",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let lock_b = ReentrantLock::from_url_with_options(
        "test_reentrant_url_contention",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;

    assert!(lock_a.try_lock().await.unwrap());
    assert!(lock_a.try_lock().await.unwrap());
    assert!(!lock_b.try_lock().await.unwrap());

    lock_a.unlock().await.unwrap();
    assert!(!lock_b.try_lock().await.unwrap());

    lock_a.unlock().await.unwrap();
    assert!(lock_b.try_lock().await.unwrap());
    lock_b.unlock().await.unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
}

#[tokio::test]
async fn test_try_lock_timeout_returns_false_without_hanging() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let lock_a = match ReentrantLock::from_url_with_options(
        "test_reentrant_try_timeout",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let lock_b = ReentrantLock::from_url_with_options(
        "test_reentrant_try_timeout",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;

    lock_a.lock().await.unwrap();
    let started = Instant::now();
    assert!(!lock_b
        .try_lock_timeout(Duration::from_millis(180))
        .await
        .unwrap());
    assert!(started.elapsed() < Duration::from_secs(1));

    lock_a.unlock().await.unwrap();
    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
}

#[tokio::test]
async fn test_reentrant_lock_expires_without_watchdog() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_millis(250))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let lock_a = match ReentrantLock::from_url_with_options(
        "test_reentrant_ttl_expiry",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let lock_b = ReentrantLock::from_url_with_options(
        "test_reentrant_ttl_expiry",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;

    lock_a.lock().await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(lock_b.try_lock().await.unwrap());
    lock_b.unlock().await.unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_watchdog_keeps_reentrant_lock_alive_past_ttl() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_millis(300))
        .with_watchdog_interval(Duration::from_millis(100))
        .with_watchdog(true);
    let lock_a = match ReentrantLock::from_url_with_options(
        "test_reentrant_watchdog_renews",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let lock_b = ReentrantLock::from_url_with_options(
        "test_reentrant_watchdog_renews",
        &redis_url,
        options.with_watchdog(false),
    )
    .await
    .unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;

    lock_a.lock().await.unwrap();
    tokio::time::sleep(Duration::from_millis(800)).await;

    assert!(lock_a.is_locked().await.unwrap());
    assert!(!lock_b.try_lock().await.unwrap());

    lock_a.unlock().await.unwrap();
    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
}

#[tokio::test]
async fn test_reentrant_lock_blocking() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock = ReentrantLock::new(
        "test_blocking_lock",
        client.clone(),
        LockOptions::new()
            .with_ttl(Duration::from_secs(5))
            .with_acquire_timeout(Duration::from_secs(2)),
    );

    // 清理
    let _ = lock.force_unlock().await;

    // 获取锁
    lock.lock().await.unwrap();
    assert!(lock.is_locked().await.unwrap());

    // 释放
    lock.unlock().await.unwrap();

    // 清理
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_fair_lock() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock = FairLock::new(
        "test_fair_lock",
        client.clone(),
        LockOptions::new().with_ttl(Duration::from_secs(5)),
    );

    // 清理
    let _ = lock.force_unlock().await;

    // 尝试获取公平锁
    let result = lock.try_lock().await.unwrap();
    assert!(result, "Should acquire fair lock");

    // 释放
    lock.unlock().await.unwrap();

    // 清理
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_readwrite_lock() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let rwlock = ReadWriteLock::new(
        "test_rw_lock",
        client.clone(),
        LockOptions::new().with_ttl(Duration::from_secs(5)),
    );

    let read_lock = rwlock.read_lock();
    let write_lock = rwlock.write_lock();

    // 清理
    let _ = read_lock.force_unlock().await;
    let _ = write_lock.force_unlock().await;

    // 测试读锁（可以多个并发）
    read_lock.lock().await.unwrap();
    assert!(read_lock.is_locked().await.unwrap());

    // 释放读锁
    read_lock.unlock().await.unwrap();

    // 测试写锁
    write_lock.lock().await.unwrap();
    assert!(write_lock.is_locked().await.unwrap());

    // 释放写锁
    write_lock.unlock().await.unwrap();

    // 清理
    let _ = read_lock.force_unlock().await;
    let _ = write_lock.force_unlock().await;
}

#[tokio::test]
async fn test_readwrite_lock_exclusion() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let rwlock = ReadWriteLock::new(
        "test_rw_lock_exclusion",
        client.clone(),
        LockOptions::new()
            .with_ttl(Duration::from_secs(5))
            .with_watchdog(false),
    );

    let read_lock = rwlock.read_lock();
    let write_lock = rwlock.write_lock();

    let _ = read_lock.force_unlock().await;
    let _ = write_lock.force_unlock().await;

    read_lock.lock().await.unwrap();
    assert!(
        !write_lock.try_lock().await.unwrap(),
        "writer must not acquire while a reader is active"
    );
    read_lock.unlock().await.unwrap();

    write_lock.lock().await.unwrap();
    assert!(
        !read_lock.try_lock().await.unwrap(),
        "reader must not acquire while a writer is active"
    );
    write_lock.unlock().await.unwrap();

    let _ = read_lock.force_unlock().await;
    let _ = write_lock.force_unlock().await;
}

#[tokio::test]
async fn test_readwrite_multiple_readers_block_writer_until_all_release() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let rwlock_a = match ReadWriteLock::from_url_with_options(
        "test_rw_multiple_readers",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let rwlock_b = ReadWriteLock::from_url_with_options(
        "test_rw_multiple_readers",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let read_a = rwlock_a.read_lock();
    let read_b = rwlock_b.read_lock();
    let writer = rwlock_a.write_lock();

    let _ = read_a.force_unlock().await;
    let _ = writer.force_unlock().await;

    assert!(read_a.try_lock().await.unwrap());
    assert!(read_b.try_lock().await.unwrap());
    assert!(!writer.try_lock().await.unwrap());

    read_a.unlock().await.unwrap();
    assert!(!writer.try_lock().await.unwrap());

    read_b.unlock().await.unwrap();
    assert!(writer.try_lock().await.unwrap());
    writer.unlock().await.unwrap();

    let _ = read_a.force_unlock().await;
    let _ = writer.force_unlock().await;
}

#[tokio::test]
async fn test_write_lock_reentrant_requires_matching_unlocks() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let rwlock_a = match ReadWriteLock::from_url_with_options(
        "test_rw_write_reentrant",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let rwlock_b = ReadWriteLock::from_url_with_options(
        "test_rw_write_reentrant",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let writer = rwlock_a.write_lock();
    let other_reader = rwlock_b.read_lock();

    let _ = writer.force_unlock().await;
    let _ = other_reader.force_unlock().await;

    assert!(writer.try_lock().await.unwrap());
    assert!(writer.try_lock().await.unwrap());
    writer.unlock().await.unwrap();

    assert!(writer.is_locked().await.unwrap());
    assert!(!other_reader.try_lock().await.unwrap());

    writer.unlock().await.unwrap();
    assert!(other_reader.try_lock().await.unwrap());
    other_reader.unlock().await.unwrap();

    let _ = writer.force_unlock().await;
    let _ = other_reader.force_unlock().await;
}

#[tokio::test]
async fn test_fair_lock_failed_try_lock_does_not_leave_queue_entry() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock_a = FairLock::new(
        "test_fair_lock_queue_cleanup",
        client.clone(),
        LockOptions::new()
            .with_ttl(Duration::from_secs(5))
            .with_watchdog(false),
    );
    let lock_b = FairLock::new(
        "test_fair_lock_queue_cleanup",
        client.clone(),
        LockOptions::new()
            .with_ttl(Duration::from_secs(5))
            .with_watchdog(false),
    );

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;

    lock_a.lock().await.unwrap();
    assert!(!lock_b.try_lock().await.unwrap());
    assert!(!lock_b.try_lock().await.unwrap());
    lock_a.unlock().await.unwrap();

    assert!(lock_b.try_lock().await.unwrap());
    lock_b.unlock().await.unwrap();

    assert!(
        lock_a.try_lock().await.unwrap(),
        "failed try_lock calls must not leave stale fair queue entries"
    );
    lock_a.unlock().await.unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
}

#[tokio::test]
async fn test_fair_lock_timeout_removes_waiter_from_queue() {
    let redis_url = test_redis_url();
    let options = LockOptions::new()
        .with_ttl(Duration::from_secs(5))
        .with_retry_interval(Duration::from_millis(20))
        .with_watchdog(false);
    let lock_a = match FairLock::from_url_with_options(
        "test_fair_lock_timeout_cleanup",
        &redis_url,
        options.clone(),
    )
    .await
    {
        Ok(lock) => lock,
        Err(err) => {
            println!("Warning: Redis server not available at {redis_url}, skipping test: {err}");
            return;
        }
    };
    let lock_b = FairLock::from_url_with_options(
        "test_fair_lock_timeout_cleanup",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();
    let lock_c = FairLock::from_url_with_options(
        "test_fair_lock_timeout_cleanup",
        &redis_url,
        options.clone(),
    )
    .await
    .unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
    let _ = lock_c.force_unlock().await;

    lock_a.lock().await.unwrap();
    assert!(!lock_b
        .try_lock_timeout(Duration::from_millis(150))
        .await
        .unwrap());
    lock_a.unlock().await.unwrap();

    assert!(lock_c.try_lock().await.unwrap());
    lock_c.unlock().await.unwrap();

    assert!(lock_b.try_lock().await.unwrap());
    lock_b.unlock().await.unwrap();

    let _ = lock_a.force_unlock().await;
    let _ = lock_b.force_unlock().await;
    let _ = lock_c.force_unlock().await;
}

#[tokio::test]
async fn test_lock_builder() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    // 使用 builder 模式创建锁
    let lock = redis_lock::LockBuilder::new(client.clone())
        .ttl(Duration::from_secs(10))
        .acquire_timeout(Duration::from_secs(3))
        .build_reentrant("test_builder_lock");

    // 清理
    let _ = lock.force_unlock().await;

    // 测试
    let result = lock.try_lock().await.unwrap();
    assert!(result);

    lock.unlock().await.unwrap();

    // 清理
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_lock_remaining_ttl() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock = ReentrantLock::new(
        "test_ttl_lock",
        client.clone(),
        LockOptions::new().with_ttl(Duration::from_secs(5)),
    );

    // 清理
    let _ = lock.force_unlock().await;

    lock.lock().await.unwrap();

    // 检查 TTL
    let ttl = lock.remaining_ttl().await.unwrap();
    assert!(ttl.is_some());
    let ttl_duration = ttl.unwrap();
    assert!(ttl_duration.as_secs() > 0 && ttl_duration.as_secs() <= 5);

    lock.unlock().await.unwrap();

    // 清理
    let _ = lock.force_unlock().await;
}

#[tokio::test]
async fn test_force_unlock() {
    let client = match create_test_client().await {
        Some(c) => c,
        None => return,
    };

    let lock = ReentrantLock::new(
        "test_force_unlock",
        client.clone(),
        LockOptions::new().with_ttl(Duration::from_secs(30)),
    );

    lock.lock().await.unwrap();
    assert!(lock.is_locked().await.unwrap());

    // 强制释放
    lock.force_unlock().await.unwrap();
    assert!(!lock.is_locked().await.unwrap());
}
