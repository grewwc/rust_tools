use crate::client::RedisLockClient;
use crate::error::Result;
use crate::lock::{Lock, LockOptions};
use crate::watchdog::Watchdog;
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Mutex;
use uuid::Uuid;

/// 读写锁，类似 Redisson 的 RReadWriteLock
/// 支持多个读者同时访问，写者互斥访问
pub struct ReadWriteLock {
    read_lock: ReadLock,
    write_lock: WriteLock,
}

impl ReadWriteLock {
    pub fn new(
        name: &str,
        client: Arc<RedisLockClient>,
        options: crate::lock::LockOptions,
    ) -> Self {
        let read_lock = ReadLock::new(name, client.clone(), options.clone());
        let write_lock = WriteLock::new(name, client.clone(), options.clone());

        Self {
            read_lock,
            write_lock,
        }
    }

    /// 通过 Redis URL 直接创建读写锁。
    pub async fn from_url(name: &str, redis_url: &str) -> Result<Self> {
        Self::from_url_with_options(name, redis_url, LockOptions::default()).await
    }

    /// 通过 Redis URL 和自定义选项直接创建读写锁。
    pub async fn from_url_with_options(
        name: &str,
        redis_url: &str,
        options: LockOptions,
    ) -> Result<Self> {
        let client = Arc::new(RedisLockClient::from_url(redis_url).await?);
        Ok(Self::new(name, client, options))
    }

    /// 获取读锁
    pub fn read_lock(&self) -> &ReadLock {
        &self.read_lock
    }

    /// 获取写锁
    pub fn write_lock(&self) -> &WriteLock {
        &self.write_lock
    }
}

/// 读锁
pub struct ReadLock {
    name: String,
    client: Arc<RedisLockClient>,
    options: crate::lock::LockOptions,
    owner_id: String,
    watchdog: Mutex<Option<Watchdog>>,
}

impl ReadLock {
    pub fn new(
        name: &str,
        client: Arc<RedisLockClient>,
        options: crate::lock::LockOptions,
    ) -> Self {
        let owner_id = format!("{:?}:{}", std::thread::current().id(), Uuid::new_v4());

        Self {
            name: name.to_string(),
            client,
            options,
            owner_id,
            watchdog: Mutex::new(None),
        }
    }

    fn get_lock_key(&self) -> String {
        format!("redis_lock:rw:{}:read", self.name)
    }

    fn get_state_key(&self) -> String {
        format!("redis_lock:rw:{}:state", self.name)
    }

    fn get_write_key(&self) -> String {
        format!("redis_lock:rw:{}:write", self.name)
    }

    fn get_try_lock_script() -> &'static str {
        r#"
        local read_key = KEYS[1]
        local write_key = KEYS[2]
        local state_key = KEYS[3]
        local owner_id = ARGV[1]
        local ttl = tonumber(ARGV[2])
        
        -- 检查是否有写锁
        local write_locked = redis.call('EXISTS', write_key)
        if write_locked == 1 then
            return 0  -- 有写锁，不能获取读锁
        end
        
        -- 增加读锁计数
        local read_count = redis.call('HINCRBY', read_key, owner_id, 1)
        redis.call('PEXPIRE', read_key, ttl)
        
        -- 标记有读锁
        redis.call('HINCRBY', state_key, 'read_count', 1)
        redis.call('PEXPIRE', state_key, ttl)
        
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

        let mut watchdog = Watchdog::for_hash_field(
            self.get_lock_key(),
            self.get_state_key(),
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
impl Lock for ReadLock {
    async fn try_lock(&self) -> Result<bool> {
        let state_key = self.get_state_key();
        let read_key = self.get_lock_key();
        let write_key = self.get_write_key();

        let script = Self::get_try_lock_script();
        let result = self
            .client
            .eval_script_i64(
                script,
                &[&read_key, &write_key, &state_key],
                &[&self.owner_id, &self.options.ttl.as_millis().to_string()],
            )
            .await?;

        if result == 1 {
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
                    "Failed to acquire read lock within timeout".to_string(),
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
        let read_key = self.get_lock_key();
        let state_key = self.get_state_key();

        let script = r#"
        local read_key = KEYS[1]
        local state_key = KEYS[2]
        local owner_id = ARGV[1]
        
        local count = redis.call('HGET', read_key, owner_id)
        if not count then
            return -1  -- 没有持有读锁
        end
        
        local new_count = tonumber(count) - 1
        if new_count <= 0 then
            redis.call('HDEL', read_key, owner_id)
        else
            redis.call('HSET', read_key, owner_id, new_count)
        end
        
        local remaining = redis.call('HLEN', read_key)
        if remaining == 0 then
            redis.call('DEL', read_key)
            redis.call('DEL', state_key)
        else
            redis.call('HSET', state_key, 'read_count', remaining)
            local remaining_ttl = redis.call('PTTL', read_key)
            if remaining_ttl > 0 then
                redis.call('PEXPIRE', state_key, remaining_ttl)
            end
        end
        
        if new_count <= 0 then
            return 1
        end
        return 2
        "#;

        let result = self
            .client
            .eval_script_i64(script, &[&read_key, &state_key], &[&self.owner_id])
            .await?;

        match result {
            1 => {
                self.stop_watchdog().await;
                Ok(())
            }
            2 => Ok(()),
            -1 => Err(crate::error::Error::ReleaseFailed(
                "Not holding read lock".to_string(),
            )),
            _ => Err(crate::error::Error::ReleaseFailed(
                "Failed to release read lock".to_string(),
            )),
        }
    }

    async fn is_locked(&self) -> Result<bool> {
        let read_key = self.get_lock_key();
        let exists = self.client.exists(&read_key).await?;
        Ok(exists)
    }

    async fn remaining_ttl(&self) -> Result<Option<Duration>> {
        let read_key = self.get_lock_key();
        let ttl = self.client.pttl(&read_key).await?;
        if ttl < 0 {
            Ok(None)
        } else {
            Ok(Some(Duration::from_millis(ttl as u64)))
        }
    }

    async fn force_unlock(&self) -> Result<()> {
        let read_key = self.get_lock_key();
        let state_key = self.get_state_key();
        self.client.del(&read_key).await?;
        self.client.del(&state_key).await?;
        self.stop_watchdog().await;
        Ok(())
    }
}

unsafe impl Send for ReadLock {}
unsafe impl Sync for ReadLock {}

/// 写锁
pub struct WriteLock {
    name: String,
    client: Arc<RedisLockClient>,
    options: crate::lock::LockOptions,
    owner_id: String,
    lock_id: String,
    watchdog: Mutex<Option<Watchdog>>,
}

impl WriteLock {
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
            lock_id,
            watchdog: Mutex::new(None),
        }
    }

    fn get_lock_key(&self) -> String {
        format!("redis_lock:rw:{}:write", self.name)
    }

    fn get_state_key(&self) -> String {
        format!("redis_lock:rw:{}:state", self.name)
    }

    fn get_read_key(&self) -> String {
        format!("redis_lock:rw:{}:read", self.name)
    }

    fn get_try_lock_script() -> &'static str {
        r#"
        local write_key = KEYS[1]
        local read_key = KEYS[2]
        local state_key = KEYS[3]
        local lock_id = ARGV[1]
        local owner_id = ARGV[2]
        local ttl = tonumber(ARGV[3])
        local now = tonumber(ARGV[4])

        local existing = redis.call('GET', write_key)
        if existing then
            local lock_data = cjson.decode(existing)
            if lock_data.owner == owner_id then
                lock_data.reentrant_count = (lock_data.reentrant_count or 1) + 1
                lock_data.expire_at = now + ttl
                redis.call('SET', write_key, cjson.encode(lock_data), 'PX', ttl)
                redis.call('PEXPIRE', state_key, ttl)
                return 1
            end
            return 0
        end
        
        -- 检查是否有读锁
        if redis.call('EXISTS', read_key) == 1 then
            return 0  -- 有锁存在，不能获取写锁
        end
        
        -- 获取写锁
        redis.call('SET', write_key, cjson.encode({
            id = lock_id,
            owner = owner_id,
            reentrant_count = 1,
            expire_at = now + ttl
        }), 'PX', ttl)
        
        redis.call('HSET', state_key, 'write_locked', 1)
        redis.call('PEXPIRE', state_key, ttl)
        
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
            vec![self.get_lock_key(), self.get_state_key()],
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
impl Lock for WriteLock {
    async fn try_lock(&self) -> Result<bool> {
        let state_key = self.get_state_key();
        let write_key = self.get_lock_key();
        let read_key = self.get_read_key();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let script = Self::get_try_lock_script();
        let result = self
            .client
            .eval_script_i64(
                script,
                &[&write_key, &read_key, &state_key],
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
                    "Failed to acquire write lock within timeout".to_string(),
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
        let write_key = self.get_lock_key();
        let state_key = self.get_state_key();

        let script = r#"
        local write_key = KEYS[1]
        local state_key = KEYS[2]
        local owner_id = ARGV[1]
        
        local existing = redis.call('GET', write_key)
        if not existing then
            return 0  -- 锁不存在
        end
        
        local lock_data = cjson.decode(existing)
        if lock_data.owner ~= owner_id then
            return -1  -- 不是持有者
        end

        lock_data.reentrant_count = (lock_data.reentrant_count or 1) - 1
        if lock_data.reentrant_count > 0 then
            local remaining_ttl = redis.call('PTTL', write_key)
            if remaining_ttl <= 0 then
                return 0
            end
            redis.call('SET', write_key, cjson.encode(lock_data), 'PX', remaining_ttl)
            redis.call('PEXPIRE', state_key, remaining_ttl)
            return 2
        end
        
        redis.call('DEL', write_key)
        redis.call('DEL', state_key)
        
        return 1
        "#;

        let result = self
            .client
            .eval_script_i64(script, &[&write_key, &state_key], &[&self.owner_id])
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
        let write_key = self.get_lock_key();
        self.client.exists(&write_key).await
    }

    async fn remaining_ttl(&self) -> Result<Option<Duration>> {
        let write_key = self.get_lock_key();
        let ttl = self.client.pttl(&write_key).await?;
        if ttl < 0 {
            Ok(None)
        } else {
            Ok(Some(Duration::from_millis(ttl as u64)))
        }
    }

    async fn force_unlock(&self) -> Result<()> {
        let write_key = self.get_lock_key();
        let state_key = self.get_state_key();
        self.client.del(&write_key).await?;
        self.client.del(&state_key).await?;
        self.stop_watchdog().await;
        Ok(())
    }
}

unsafe impl Send for WriteLock {}
unsafe impl Sync for WriteLock {}
