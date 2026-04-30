use crate::client::RedisLockClient;
use crate::error::Result;
use crate::lock::{Lock, LockOptions};
use crate::watchdog::Watchdog;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use uuid::Uuid;

/// 公平锁，按照请求顺序获取锁（FIFO）
/// 使用 Redis 的 list 结构实现队列
pub struct FairLock {
    /// 锁的名称
    name: String,
    /// Redis 客户端
    client: Arc<RedisLockClient>,
    /// 锁的配置选项
    options: crate::lock::LockOptions,
    /// 锁的唯一标识
    lock_id: String,
    /// 等待队列的 key
    queue_key: String,
    /// 锁持有者标识
    owner_id: String,
    watchdog: Mutex<Option<Watchdog>>,
}

impl FairLock {
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
            lock_id,
            queue_key: format!("redis_lock:fair:{}:queue", name),
            owner_id,
            watchdog: Mutex::new(None),
        }
    }

    /// 通过 Redis URL 直接创建公平锁。
    pub async fn from_url(name: &str, redis_url: &str) -> Result<Self> {
        Self::from_url_with_options(name, redis_url, LockOptions::default()).await
    }

    /// 通过 Redis URL 和自定义选项直接创建公平锁。
    pub async fn from_url_with_options(
        name: &str,
        redis_url: &str,
        options: LockOptions,
    ) -> Result<Self> {
        let client = Arc::new(RedisLockClient::from_url(redis_url).await?);
        Ok(Self::new(name, client, options))
    }

    fn get_lock_key(&self) -> String {
        format!("redis_lock:fair:{}", self.name)
    }

    /// 获取公平锁的 Lua 脚本（使用队列实现 FIFO）
    fn get_try_lock_script() -> &'static str {
        r#"
        local lock_key = KEYS[1]
        local queue_key = KEYS[2]
        local lock_id = ARGV[1]
        local owner_id = ARGV[2]
        local ttl = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local existing = redis.call('GET', lock_key)
        if existing then
            local lock_data = cjson.decode(existing)
            if lock_data.owner == owner_id then
                lock_data.reentrant_count = (lock_data.reentrant_count or 1) + 1
                lock_data.expire_at = now + ttl
                redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', ttl)
                redis.call('LREM', queue_key, 0, owner_id)
                return 1
            end
        end
        
        -- 将请求加入队列，同一 owner 只保留一个等待项
        local queued = 0
        local waiters = redis.call('LRANGE', queue_key, 0, -1)
        for _, waiter in ipairs(waiters) do
            if waiter == owner_id then
                queued = 1
                break
            end
        end
        if queued == 0 then
            redis.call('RPUSH', queue_key, owner_id)
        end
        redis.call('EXPIRE', queue_key, math.ceil(ttl/1000) + 10)
        
        -- 检查是否是队首元素
        local first = redis.call('LINDEX', queue_key, 0)
        if first ~= owner_id then
            return 0  -- 不是队首，不能获取锁
        end
        
        if not existing then
            -- 锁不存在，获取锁
            redis.call('SET', lock_key, cjson.encode({
                id = lock_id,
                owner = owner_id,
                reentrant_count = 1,
                expire_at = now + ttl
            }), 'PX', ttl)
            redis.call('LPOP', queue_key)
            return 1
        end
        
        -- 锁已被持有
        return 0
        "#
    }

    fn get_unlock_script() -> &'static str {
        r#"
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        
        local existing = redis.call('GET', lock_key)
        if not existing then
            return 0
        end
        
        local lock_data = cjson.decode(existing)
        if lock_data.owner ~= owner_id then
            return -1
        end

        lock_data.reentrant_count = (lock_data.reentrant_count or 1) - 1
        if lock_data.reentrant_count > 0 then
            local remaining_ttl = redis.call('PTTL', lock_key)
            if remaining_ttl <= 0 then
                return 0
            end
            redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', remaining_ttl)
            return 2
        end
        
        redis.call('DEL', lock_key)
        return 1
        "#
    }

    fn get_remove_queue_script() -> &'static str {
        r#"
        local queue_key = KEYS[1]
        local owner_id = ARGV[1]
        redis.call('LREM', queue_key, 0, owner_id)
        return 1
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
            vec![self.get_lock_key()],
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

    async fn remove_queue_entry(&self) -> Result<()> {
        self.client
            .eval_script_i64(
                Self::get_remove_queue_script(),
                &[&self.queue_key],
                &[&self.owner_id],
            )
            .await?;
        Ok(())
    }

    async fn try_lock_queued(&self, keep_waiter_on_failure: bool) -> Result<bool> {
        let lock_key = self.get_lock_key();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let script = Self::get_try_lock_script();
        let result = self
            .client
            .eval_script_i64(
                script,
                &[&lock_key, &self.queue_key],
                &[
                    &self.lock_id,
                    &self.owner_id,
                    &self.options.ttl.as_millis().to_string(),
                    &now.to_string(),
                ],
            )
            .await?;

        if result == 1 {
            self.start_watchdog().await;
            Ok(true)
        } else {
            if !keep_waiter_on_failure {
                self.remove_queue_entry().await?;
            }
            Ok(false)
        }
    }
}

#[async_trait]
impl Lock for FairLock {
    async fn try_lock(&self) -> Result<bool> {
        self.try_lock_queued(false).await
    }

    async fn lock(&self) -> Result<()> {
        let start = std::time::Instant::now();
        loop {
            if self.try_lock_queued(true).await? {
                return Ok(());
            }

            if start.elapsed() > self.options.acquire_timeout {
                self.remove_queue_entry().await?;
                return Err(crate::error::Error::AcquireFailed(
                    "Failed to acquire fair lock within timeout".to_string(),
                ));
            }

            tokio::time::sleep(self.options.retry_interval).await;
        }
    }

    async fn try_lock_timeout(&self, timeout: Duration) -> Result<bool> {
        let start = std::time::Instant::now();
        loop {
            if self.try_lock_queued(true).await? {
                return Ok(true);
            }

            if start.elapsed() > timeout {
                self.remove_queue_entry().await?;
                return Ok(false);
            }

            tokio::time::sleep(self.options.retry_interval).await;
        }
    }

    async fn unlock(&self) -> Result<()> {
        let lock_key = self.get_lock_key();
        let script = Self::get_unlock_script();
        let result = self
            .client
            .eval_script_i64(script, &[&lock_key], &[&self.owner_id])
            .await?;

        match result {
            1 => {
                self.stop_watchdog().await;
                Ok(())
            }
            2 => Ok(()),
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
        self.client.del(&lock_key).await?;
        self.client.del(&self.queue_key).await?;
        self.stop_watchdog().await;
        Ok(())
    }
}

unsafe impl Send for FairLock {}
unsafe impl Sync for FairLock {}
