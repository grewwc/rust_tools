//! 轻量级密钥加密/解密模块
//!
//! 用于保护 models.json 中的 api_key 等敏感信息。
//!
//! # 设计
//! - 密钥文件：`~/.configW/.secret`（32 字节随机数，base64 编码，chmod 0600）
//! - 加密格式：`enc:<base64(nonce_8bytes + ciphertext)>`
//! - 算法：XOR stream cipher，密钥流由 SHA-256(secret + nonce) 生成
//!
//! # 安全模型
//! - 依赖文件权限（0600）保护密钥
//! - 每次加密使用随机 nonce，相同明文产生不同密文
//! - 适合"静态数据保护"场景，非对抗性密码学方案

use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use rand::Rng;
use sha2::{Digest, Sha256};

use super::configw::config_path;

/// 密钥文件路径：`~/.configW.secret`
///
/// 与配置文件 `~/.configW` 同级存放，文件名以 `.` 开头隐藏。
fn secret_path() -> PathBuf {
    let config = config_path();
    let parent = config.parent().unwrap_or(config.as_ref());
    parent.join(".configW.secret")
}

/// 读取或生成密钥（32 字节）
fn load_or_create_secret() -> io::Result<Vec<u8>> {
    let path = secret_path();

    // 尝试读取现有密钥
    if path.exists() {
        let content = fs::read_to_string(&path)?;
        let trimmed = content.trim();
        if let Ok(key) = B64.decode(trimmed) {
            if key.len() == 32 {
                return Ok(key);
            }
        }
        // 文件存在但格式不对，报错
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "secret file {} has invalid format (expected 32-byte base64)",
                path.display()
            ),
        ));
    }

    // 生成新密钥
    let mut key = vec![0u8; 32];
    rand::rng().fill_bytes(&mut key);
    let encoded = B64.encode(&key);

    // 确保父目录存在（通常 ~/.configW 同级就是 home 目录）
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // 写入文件并设置权限 0600
    fs::write(&path, &encoded)?;
    let mut perms = fs::metadata(&path)?.permissions();
    perms.set_mode(0o600);
    fs::set_permissions(&path, perms)?;

    eprintln!("[secret] generated new key at {}", path.display());
    Ok(key)
}

/// 生成 SHA-256(secret + nonce) 作为密钥流
fn derive_keystream(secret: &[u8], nonce: &[u8], len: usize) -> Vec<u8> {
    let mut stream = Vec::with_capacity(len);
    let mut counter = 0u32;

    while stream.len() < len {
        let mut hasher = Sha256::new();
        hasher.update(secret);
        hasher.update(nonce);
        hasher.update(counter.to_le_bytes());
        let hash = hasher.finalize();
        stream.extend_from_slice(&hash);
        counter += 1;
    }

    stream.truncate(len);
    stream
}

/// 加密明文，返回 `enc:<base64>` 格式字符串
pub fn encrypt(plaintext: &str) -> io::Result<String> {
    let secret = load_or_create_secret()?;

    // 生成 8 字节随机 nonce
    let mut nonce = [0u8; 8];
    rand::rng().fill_bytes(&mut nonce);

    // 生成密钥流并 XOR
    let keystream = derive_keystream(&secret, &nonce, plaintext.len());
    let ciphertext: Vec<u8> = plaintext
        .bytes()
        .zip(keystream.iter())
        .map(|(p, k)| p ^ k)
        .collect();

    // 拼接 nonce + ciphertext
    let mut payload = Vec::with_capacity(nonce.len() + ciphertext.len());
    payload.extend_from_slice(&nonce);
    payload.extend_from_slice(&ciphertext);

    Ok(format!("enc:{}", B64.encode(&payload)))
}

/// 解密 `enc:<base64>` 格式字符串，返回明文
pub fn decrypt(encoded: &str) -> io::Result<String> {
    let payload_b64 = encoded
        .strip_prefix("enc:")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing 'enc:' prefix"))?;

    let payload = B64.decode(payload_b64).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("base64 decode error: {e}"),
        )
    })?;

    if payload.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "payload too short (expected at least 8 bytes for nonce)",
        ));
    }

    let (nonce, ciphertext) = payload.split_at(8);
    let secret = load_or_create_secret()?;

    // 生成密钥流并 XOR
    let keystream = derive_keystream(&secret, nonce, ciphertext.len());
    let plaintext: Vec<u8> = ciphertext
        .iter()
        .zip(keystream.iter())
        .map(|(c, k)| c ^ k)
        .collect();

    String::from_utf8(plaintext)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("invalid UTF-8: {e}")))
}

/// 判断字符串是否为加密格式
pub fn is_encrypted(s: &str) -> bool {
    s.starts_with("enc:")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    fn make_temp_config_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("configw_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_roundtrip() {
        let dir = make_temp_config_dir();
        // SAFETY: 测试中单独设置环境变量，不影响其他测试
        unsafe { env::set_var("CONFIGW_PATH", dir.to_str().unwrap()) };

        let plaintext = "my-secret-api-key-12345";
        let encrypted = encrypt(plaintext).unwrap();

        assert!(encrypted.starts_with("enc:"));
        assert_ne!(encrypted, format!("enc:{}", plaintext));

        let decrypted = decrypt(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);

        // 每次加密产生不同密文（因为 nonce 随机）
        let encrypted2 = encrypt(plaintext).unwrap();
        assert_ne!(encrypted, encrypted2);

        let decrypted2 = decrypt(&encrypted2).unwrap();
        assert_eq!(decrypted2, plaintext);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_is_encrypted() {
        assert!(is_encrypted("enc:abc123"));
        assert!(!is_encrypted("plain-key"));
        assert!(!is_encrypted("enc"));
    }

    #[test]
    fn test_decrypt_invalid_format() {
        assert!(decrypt("no-prefix").is_err());
        assert!(decrypt("enc:").is_err());
        assert!(decrypt("enc:not-valid-base64!!!").is_err());
    }
}
