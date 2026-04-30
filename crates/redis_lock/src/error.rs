use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Lock not found: {0}")]
    LockNotFound(String),

    #[error("Lock already held: {0}")]
    LockAlreadyHeld(String),

    #[error("Failed to acquire lock: {0}")]
    AcquireFailed(String),

    #[error("Failed to release lock: {0}")]
    ReleaseFailed(String),

    #[error("Lock expired: {0}")]
    LockExpired(String),

    #[error("Watchdog error: {0}")]
    Watchdog(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;
