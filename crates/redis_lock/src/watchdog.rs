use crate::client::RedisLockClient;
use crate::error::{Error, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task::JoinHandle;
use tokio::time;

#[derive(Clone)]
enum WatchdogTarget {
    JsonLock {
        keys: Vec<String>,
        owner_id: String,
    },
    HashField {
        read_key: String,
        state_key: String,
        owner_id: String,
    },
}

impl WatchdogTarget {
    async fn renew(&self, client: &RedisLockClient, lock_ttl: Duration) -> Result<bool> {
        match self {
            Self::JsonLock { keys, owner_id } => {
                let script = r#"
                local lock_key = KEYS[1]
                local owner_id = ARGV[1]
                local ttl = tonumber(ARGV[2])
                local now = tonumber(ARGV[3])

                local existing = redis.call('GET', lock_key)
                if not existing then
                    return 0
                end

                local lock_data = cjson.decode(existing)
                if lock_data.owner ~= owner_id then
                    return -1
                end

                lock_data.expire_at = now + ttl
                redis.call('SET', lock_key, cjson.encode(lock_data), 'PX', ttl)
                for i = 2, #KEYS do
                    redis.call('PEXPIRE', KEYS[i], ttl)
                end
                return 1
                "#;
                let ttl = lock_ttl.as_millis().to_string();
                let now = current_time_millis().to_string();
                let key_refs = keys.iter().map(String::as_str).collect::<Vec<_>>();
                let result = client
                    .eval_script_i64(script, &key_refs, &[owner_id, &ttl, &now])
                    .await?;
                Ok(result == 1)
            }
            Self::HashField {
                read_key,
                state_key,
                owner_id,
            } => {
                let script = r#"
                local read_key = KEYS[1]
                local state_key = KEYS[2]
                local owner_id = ARGV[1]
                local ttl = tonumber(ARGV[2])

                local count = redis.call('HGET', read_key, owner_id)
                if not count then
                    return 0
                end

                redis.call('PEXPIRE', read_key, ttl)
                redis.call('PEXPIRE', state_key, ttl)
                return 1
                "#;
                let ttl = lock_ttl.as_millis().to_string();
                let result = client
                    .eval_script_i64(script, &[read_key, state_key], &[owner_id, &ttl])
                    .await?;
                Ok(result == 1)
            }
        }
    }
}

fn current_time_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

/// Watchdog 用于自动续期分布式锁，防止锁超时释放
/// 类似 Redisson 的 Watchdog 机制
pub struct Watchdog {
    /// Redis 客户端
    client: Arc<RedisLockClient>,
    /// 续期间隔
    interval: Duration,
    /// 锁的过期时间
    lock_ttl: Duration,
    target: WatchdogTarget,
    /// 是否正在运行
    running: Arc<AtomicBool>,
    /// 后台任务的句柄
    task_handle: Option<JoinHandle<()>>,
}

impl Watchdog {
    pub fn for_json_lock(
        keys: Vec<String>,
        client: Arc<RedisLockClient>,
        interval: Duration,
        lock_ttl: Duration,
        owner_id: &str,
    ) -> Self {
        Self {
            client,
            interval,
            lock_ttl,
            target: WatchdogTarget::JsonLock {
                keys,
                owner_id: owner_id.to_string(),
            },
            running: Arc::new(AtomicBool::new(false)),
            task_handle: None,
        }
    }

    pub fn for_hash_field(
        read_key: String,
        state_key: String,
        client: Arc<RedisLockClient>,
        interval: Duration,
        lock_ttl: Duration,
        owner_id: &str,
    ) -> Self {
        Self {
            client,
            interval,
            lock_ttl,
            target: WatchdogTarget::HashField {
                read_key,
                state_key,
                owner_id: owner_id.to_string(),
            },
            running: Arc::new(AtomicBool::new(false)),
            task_handle: None,
        }
    }

    /// 启动 Watchdog
    pub fn start(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            return;
        }

        self.running.store(true, Ordering::SeqCst);

        let client = self.client.clone();
        let interval = self.interval;
        let lock_ttl = self.lock_ttl;
        let target = self.target.clone();
        let running = self.running.clone();

        // 启动后台任务进行续期
        let handle = tokio::spawn(async move {
            let mut interval_timer = time::interval(interval);

            while running.load(Ordering::SeqCst) {
                interval_timer.tick().await;

                if !running.load(Ordering::SeqCst) {
                    break;
                }

                // 尝试续期
                match target.renew(&client, lock_ttl).await {
                    Ok(renewed) => {
                        if !renewed {
                            // 锁不存在或不属于当前持有者，停止续期
                            running.store(false, Ordering::SeqCst);
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!("Watchdog failed to renew lock: {:?}", e);
                        // 继续尝试，不立即停止
                    }
                }
            }
        });

        self.task_handle = Some(handle);
    }

    /// 停止 Watchdog
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.task_handle.take() {
            handle.abort();
        }
    }

    /// 检查 Watchdog 是否正在运行
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    pub async fn renew_once_for_test(&self) -> Result<bool> {
        if self.interval.is_zero() {
            return Err(Error::Watchdog(
                "watchdog interval cannot be zero".to_string(),
            ));
        }
        self.target.renew(&self.client, self.lock_ttl).await
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop();
    }
}
