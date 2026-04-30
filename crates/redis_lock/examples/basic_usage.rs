use redis_lock::{Lock, LockOptions, ReentrantLock};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 使用你提供的 Redis 地址
    let redis_url = "redis://127.0.0.1:6381";

    println!("Connecting to Redis at {}", redis_url);

    // 直接通过 Redis 地址创建可重入锁
    let lock = match ReentrantLock::from_url_with_options(
        "example_lock",
        redis_url,
        LockOptions::new()
            .with_ttl(Duration::from_secs(30))
            .with_watchdog(true),
    )
    .await
    {
        Ok(lock) => {
            println!("✅ Lock client initialized successfully!");
            lock
        }
        Err(e) => {
            eprintln!("❌ Failed to connect to Redis: {}", e);
            eprintln!("Please make sure Redis server is running on {}", redis_url);
            return Ok(());
        }
    };

    // 清理可能存在的旧锁
    println!("Cleaning up any existing lock...");
    let _ = lock.force_unlock().await;

    // 尝试获取锁
    println!("Trying to acquire lock...");
    match lock.try_lock().await {
        Ok(true) => {
            println!("✅ Lock acquired successfully!");

            // 检查锁状态
            match lock.is_locked().await {
                Ok(true) => println!("🔒 Lock is currently held"),
                Ok(false) => println!("🔓 Lock is not held (unexpected)"),
                Err(e) => eprintln!("Error checking lock status: {}", e),
            }

            // 获取剩余 TTL
            match lock.remaining_ttl().await {
                Ok(Some(ttl)) => println!("⏱️  Lock TTL: {:?}", ttl),
                Ok(None) => println!("⏱️  Lock has no TTL"),
                Err(e) => eprintln!("Error getting TTL: {}", e),
            }

            // 模拟业务操作
            println!("🔧 Doing some work...");
            tokio::time::sleep(Duration::from_secs(1)).await;
            println!("✅ Work completed!");

            // 可重入：同一线程可以再次获取锁
            println!("Testing reentrant lock...");
            match lock.try_lock().await {
                Ok(true) => {
                    println!("✅ Reentrant lock acquired!");
                    println!("Doing more work...");
                    tokio::time::sleep(Duration::from_millis(500)).await;

                    // 需要释放两次（因为获取了两次）
                    lock.unlock().await?;
                    println!("🔓 Reentrant lock released (1/2)");
                }
                Ok(false) => println!("❌ Failed to acquire reentrant lock"),
                Err(e) => eprintln!("Error: {}", e),
            }

            // 释放锁
            lock.unlock().await?;
            println!("🔓 Lock released successfully!");
        }
        Ok(false) => {
            println!("❌ Failed to acquire lock (already held by another process)");
        }
        Err(e) => {
            eprintln!("❌ Error: {}", e);
        }
    }

    // 测试锁已被释放
    match lock.is_locked().await {
        Ok(false) => println!("✅ Lock has been released (as expected)"),
        Ok(true) => println!("⚠️  Lock is still held (unexpected)"),
        Err(e) => eprintln!("Error checking lock: {}", e),
    }

    println!("\n🎉 Example completed successfully!");
    Ok(())
}
