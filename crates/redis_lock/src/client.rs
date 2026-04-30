use crate::error::Result;
use redis::{aio::ConnectionManager, AsyncCommands, Client as RedisClient, ConnectionInfo};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// Redis 客户端封装，支持连接池和异步操作
#[derive(Clone)]
pub struct RedisLockClient {
    conn_manager: Arc<ConnectionManager>,
}

impl RedisLockClient {
    /// 从 Redis URL 创建客户端
    pub async fn from_url(url: &str) -> Result<Self> {
        let url = normalize_redis_url(url)?;
        let client = RedisClient::open(url.as_str())?;
        let conn_manager = connect_with_timeout(client, DEFAULT_CONNECT_TIMEOUT).await?;
        Ok(Self {
            conn_manager: Arc::new(conn_manager),
        })
    }

    /// 从连接信息创建客户端
    pub async fn from_connection_info(info: &ConnectionInfo) -> Result<Self> {
        let client = RedisClient::open(info.clone())?;
        let conn_manager = connect_with_timeout(client, DEFAULT_CONNECT_TIMEOUT).await?;
        Ok(Self {
            conn_manager: Arc::new(conn_manager),
        })
    }

    /// 获取一个连接（内部使用）
    async fn get_connection(&self) -> Result<ConnectionManager> {
        Ok(self.conn_manager.as_ref().clone())
    }

    /// 执行 Lua 脚本（原子操作），返回 i64 结果
    pub async fn eval_script_i64(&self, script: &str, keys: &[&str], args: &[&str]) -> Result<i64> {
        let mut conn = self.get_connection().await?;
        let result: Option<i64> = redis::cmd("EVAL")
            .arg(script)
            .arg(keys.len())
            .arg(keys)
            .arg(args)
            .query_async(&mut conn)
            .await?;

        Ok(result.unwrap_or(0))
    }

    /// 执行 Lua 脚本，返回字符串结果
    pub async fn eval_script_string(
        &self,
        script: &str,
        keys: &[&str],
        args: &[&str],
    ) -> Result<Option<String>> {
        let mut conn = self.get_connection().await?;
        let result: Option<String> = redis::cmd("EVAL")
            .arg(script)
            .arg(keys.len())
            .arg(keys)
            .arg(args)
            .query_async(&mut conn)
            .await?;

        Ok(result)
    }

    /// 设置键值（带过期时间）
    pub async fn set_with_expiry(&self, key: &str, value: &str, ttl: Duration) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let result: redis::Value = redis::cmd("SET")
            .arg(key)
            .arg(value)
            .arg("NX")
            .arg("PX")
            .arg(ttl.as_millis() as u64)
            .query_async(&mut conn)
            .await?;

        match result {
            redis::Value::Okay => Ok(true),
            _ => Ok(false),
        }
    }

    /// 获取键值
    pub async fn get(&self, key: &str) -> Result<Option<String>> {
        let mut conn = self.get_connection().await?;
        let result: Option<String> = conn.get(key).await?;
        Ok(result)
    }

    /// 检查键是否存在
    pub async fn exists(&self, key: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let result: bool = conn.exists(key).await?;
        Ok(result)
    }

    /// 获取键的剩余存活时间（毫秒）
    pub async fn pttl(&self, key: &str) -> Result<i64> {
        let mut conn = self.get_connection().await?;
        let result: i64 = conn.pttl(key).await?;
        Ok(result)
    }

    /// 删除键
    pub async fn del(&self, key: &str) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let result: u32 = conn.del(key).await?;
        Ok(result > 0)
    }

    /// 延长键的过期时间
    pub async fn pexpire(&self, key: &str, ttl: Duration) -> Result<bool> {
        let mut conn = self.get_connection().await?;
        let ttl_ms = ttl.as_millis() as i64;
        let result: bool = conn.pexpire(key, ttl_ms).await?;
        Ok(result)
    }
}

fn normalize_redis_url(url: &str) -> Result<String> {
    let url = url.trim();
    if url.is_empty() {
        return Err(crate::error::Error::Other(
            "Redis URL cannot be empty".to_string(),
        ));
    }
    if url.contains("://") {
        Ok(url.to_string())
    } else {
        Ok(format!("redis://{url}"))
    }
}

async fn connect_with_timeout(client: RedisClient, timeout: Duration) -> Result<ConnectionManager> {
    tokio::time::timeout(timeout, ConnectionManager::new(client))
        .await
        .map_err(|_| {
            crate::error::Error::Other(format!("Timed out connecting to Redis after {:?}", timeout))
        })?
        .map_err(crate::error::Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_redis_url_accepts_bare_host_port() {
        assert_eq!(
            normalize_redis_url("127.0.0.1:6381").unwrap(),
            "redis://127.0.0.1:6381"
        );
        assert_eq!(
            normalize_redis_url("redis://127.0.0.1:6381").unwrap(),
            "redis://127.0.0.1:6381"
        );
        assert!(normalize_redis_url(" ").is_err());
    }

    #[tokio::test]
    async fn test_client_creation() {
        let url = std::env::var("REDIS_LOCK_TEST_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6381".to_string());
        match tokio::time::timeout(Duration::from_millis(500), RedisLockClient::from_url(&url))
            .await
        {
            Ok(Ok(_)) => println!("Redis client created successfully"),
            Ok(Err(err)) => println!("Redis not available at {url}, skipping: {err}"),
            Err(_) => println!("Redis connection to {url} timed out, skipping"),
        }
    }
}
