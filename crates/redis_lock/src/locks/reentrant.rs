use crate::client::RedisLockClient;
use crate::error::Result;
use crate::lock::{Lock, LockOptions};
use crate::watchdog::Watchdog;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use uuid::Uuid;

/// 可重入分布式锁，类似 Redisson 的 RLock
pub struct ReentrantLock {
    /// 锁的名称（Redis key）
    name: String,
    /// Redis 客户端
    client: Arc<RedisLockClient>,
    /// 锁的配置选项
    options: crate::lock::LockOptions,
    /// 当前线程/协程的锁持有者标识
    owner_id: String,
    /// 本地重入计数器
    local_count: Arc<AtomicU32>,
    /// 锁的唯一 ID
    lock_id: String,
    watchdog: Mutex<Option<Watchdog>>,
}

impl ReentrantLock {
    /// 创建新的可重入锁
    pub fn new(
        name: &str,
        client: Arc<RedisLockClient>,
        options: crate::lock::LockOptions,
    ) -> Self {
        let owner_id = format!("{:?}:{}", std::thread::current().id(), Uuid::new_v4());
        let lock_id = Uuid::new_v4().to_string();

        Self {
            name: name.to_string(),
            client,
            options,
            owner_id,
            local_count: Arc::new(AtomicU32::new(0)),
            lock_id,
            watchdog: Mutex::new(None),
        }
    }

    /// 通过 Redis URL 直接创建可重入锁。
    pub async fn from_url(name: &str, redis_url: &str) -> Result<Self> {
        Self::from_url_with_options(name, redis_url, LockOptions::default()).await
    }

    /// 通过 Redis URL 和自定义选项直接创建可重入锁。
    pub async fn from_url_with_options(
        name: &str,
        redis_url: &str,
        options: LockOptions,
    ) -> Result<Self> {
        let client = Arc::new(RedisLockClient::from_url(redis_url).await?);
        Ok(Self::new(name, client, options))
    }

    /// 获取锁的 Redis key
    fn get_lock_key(&self) -> String {
        format!("redis_lock:{}", self.name)
    }

    /// 获取重入计数器的 Redis key
    fn get_reentrant_key(&self) -> String {
        format!("redis_lock:{}:reentrant:{}", self.name, self.owner_id)
    }

    /// 尝试获取锁的 Lua 脚本
    fn get_try_lock_script() -> &'static str {
        r#"
        local lock_key = KEYS[1]
        local reentrant_key = KEYS[2]
        local lock_id = ARGV[1]
        local owner_id = ARGV[2]
        local ttl = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])
        
        -- 检查锁是否存在
        local existing = redis.call('GET', lock_key)
        if not existing then
            -- 锁不存在，创建新锁
            local lock_data = {
                id = lock_id,
                owner = owner_id,
                reentrant_count = 1,
                expire_at = now + ttl
            }
            redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', ttl)
            redis.call('SET', reentrant_key, 1, 'PX', ttl)
            return 1
        end
        
        -- 锁已存在，检查是否是同一持有者
        local lock_data = cjson.decode(existing)
        if lock_data.owner == owner_id then
            -- 同一持有者，增加重入计数
            lock_data.reentrant_count = lock_data.reentrant_count + 1
            lock_data.expire_at = now + ttl
            redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', ttl)
            redis.call('INCR', reentrant_key)
            redis.call('PEXPIRE', reentrant_key, ttl)
            return 1
        end
        
        -- 不是同一持有者，获取锁失败
        return 0
        "#
    }

    /// 尝试释放锁的 Lua 脚本
    fn get_unlock_script() -> &'static str {
        r#"
        local lock_key = KEYS[1]
        local reentrant_key = KEYS[2]
        local owner_id = ARGV[1]
        local now = tonumber(ARGV[2])
        
        local existing = redis.call('GET', lock_key)
        if not existing then
            return 0  -- 锁不存在
        end
        
        local lock_data = cjson.decode(existing)
        if lock_data.owner ~= owner_id then
            return -1  -- 不是锁的持有者
        end

        local remaining_ttl = redis.call('PTTL', lock_key)
        if remaining_ttl <= 0 then
            return 0
        end
        
        -- 减少重入计数
        lock_data.reentrant_count = lock_data.reentrant_count - 1
        
        if lock_data.reentrant_count <= 0 then
            -- 重入计数为0，删除锁
            redis.call('DEL', lock_key)
            redis.call('DEL', reentrant_key)
            return 1  -- 锁已释放
        else
            -- 更新锁数据
            lock_data.expire_at = now + remaining_ttl
            redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', remaining_ttl)
            redis.call('SET', reentrant_key, lock_data.reentrant_count, 'PX', remaining_ttl)
            return 2  -- 仍有重入计数
        end
        "#
    }

    fn watchdog_interval(&self) -> Duration {
        self.options.watchdog_interval.unwrap_or_else(|| {
            let millis = (self.options.ttl.as_millis() / 3)
                .max(1)
                .min(u64::MAX as u128);
            Duration::from_millis(millis as u64)
        })
    }

    async fn start_watchdog(&self) {
        if !self.options.enable_watchdog {
            return;
        }

        let mut guard = self.watchdog.lock().await;
        if guard
            .as_ref()
            .map(|watchdog| watchdog.is_running())
            .unwrap_or(false)
        {
            return;
        }

        let mut watchdog = Watchdog::for_json_lock(
            vec![self.get_lock_key(), self.get_reentrant_key()],
            self.client.clone(),
            self.watchdog_interval(),
            self.options.ttl,
            &self.owner_id,
        );
        watchdog.start();
        *guard = Some(watchdog);
    }

    async fn stop_watchdog(&self) {
        if let Some(mut watchdog) = self.watchdog.lock().await.take() {
            watchdog.stop();
        }
    }
}

#[async_trait]
impl Lock for ReentrantLock {
    async fn try_lock(&self) -> Result<bool> {
        let lock_key = self.get_lock_key();
        let reentrant_key = self.get_reentrant_key();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let script = Self::get_try_lock_script();
        let result = self
            .client
            .eval_script_i64(
                script,
                &[&lock_key, &reentrant_key],
                &[
                    &self.lock_id,
                    &self.owner_id,
                    &self.options.ttl.as_millis().to_string(),
                    &now.to_string(),
                ],
            )
            .await?;

        if result == 1 {
            self.local_count.fetch_add(1, Ordering::SeqCst);
            self.start_watchdog().await;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn lock(&self) -> Result<()> {
        let start = std::time::Instant::now();
        loop {
            if self.try_lock().await? {
                return Ok(());
            }

            if start.elapsed() > self.options.acquire_timeout {
                return Err(crate::error::Error::AcquireFailed(
                    "Failed to acquire lock within timeout".to_string(),
                ));
            }

            tokio::time::sleep(self.options.retry_interval).await;
        }
    }

    async fn try_lock_timeout(&self, timeout: Duration) -> Result<bool> {
        let start = std::time::Instant::now();
        loop {
            if self.try_lock().await? {
                return Ok(true);
            }

            if start.elapsed() > timeout {
                return Ok(false);
            }

            tokio::time::sleep(self.options.retry_interval).await;
        }
    }

    async fn unlock(&self) -> Result<()> {
        let lock_key = self.get_lock_key();
        let reentrant_key = self.get_reentrant_key();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let script = Self::get_unlock_script();
        let result = self
            .client
            .eval_script_i64(
                script,
                &[&lock_key, &reentrant_key],
                &[&self.owner_id, &now.to_string()],
            )
            .await?;

        match result {
            1 => {
                self.local_count.store(0, Ordering::SeqCst);
                self.stop_watchdog().await;
                Ok(())
            }
            2 => {
                self.local_count.fetch_sub(1, Ordering::SeqCst);
                Ok(())
            }
            -1 => Err(crate::error::Error::ReleaseFailed(
                "Not the lock owner".to_string(),
            )),
            _ => Err(crate::error::Error::ReleaseFailed(
                "Lock not found".to_string(),
            )),
        }
    }

    async fn is_locked(&self) -> Result<bool> {
        let lock_key = self.get_lock_key();
        self.client.exists(&lock_key).await
    }

    async fn remaining_ttl(&self) -> Result<Option<Duration>> {
        let lock_key = self.get_lock_key();
        let ttl = self.client.pttl(&lock_key).await?;
        if ttl < 0 {
            Ok(None)
        } else {
            Ok(Some(Duration::from_millis(ttl as u64)))
        }
    }

    async fn force_unlock(&self) -> Result<()> {
        let lock_key = self.get_lock_key();
        let reentrant_key = self.get_reentrant_key();
        self.client.del(&lock_key).await?;
        self.client.del(&reentrant_key).await?;
        self.local_count.store(0, Ordering::SeqCst);
        self.stop_watchdog().await;
        Ok(())
    }
}

unsafe impl Send for ReentrantLock {}
unsafe impl Sync for ReentrantLock {}
