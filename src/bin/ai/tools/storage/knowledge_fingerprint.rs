/// 知识缓存的指纹验证模块
/// 
/// 解决"缓存未过期但实际知识已变化"的问题
/// 
/// 策略：
/// 1. 缓存时记录相关文件的指纹（hash/mtime）
/// 2. 使用前验证指纹是否匹配
/// 3. 指纹不匹配 → 即使 TTL 未过期也刷新

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, BufReader};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

/// 文件指纹信息
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileFingerprint {
    /// 文件路径（相对路径）
    pub path: String,
    /// 文件大小
    pub size: u64,
    /// 最后修改时间（Unix 时间戳）
    pub mtime: u64,
    /// 文件内容 hash（可选，用于关键文件）
    pub hash: Option<String>,
}

/// 知识指纹（用于验证缓存有效性）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeFingerprint {
    /// 关联的文件列表
    pub files: Vec<FileFingerprint>,
    /// 指纹生成时间
    pub created_at: u64,
    /// 关联的上下文（如项目路径）
    pub context: HashMap<String, String>,
    /// Git 提交 hash（如果在 Git 仓库中）
    pub git_commit: Option<String>,
}

impl KnowledgeFingerprint {
    /// 创建新的指纹
    pub fn new(context: &HashMap<String, String>) -> Self {
        Self {
            files: Vec::new(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            context: context.clone(),
            git_commit: None,
        }
    }
    
    /// 添加文件指纹
    pub fn add_file(&mut self, path: &Path, include_hash: bool) -> Result<(), String> {
        let metadata = fs::metadata(path)
            .map_err(|e| format!("Failed to get metadata for {:?}: {}", path, e))?;
        
        let mtime = metadata.modified()
            .map_err(|e| format!("Failed to get mtime: {}", e))?
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        let hash = if include_hash {
            Some(self::compute_file_hash(path)?)
        } else {
            None
        };
        
        self.files.push(FileFingerprint {
            path: path.to_string_lossy().to_string(),
            size: metadata.len(),
            mtime,
            hash,
        });
        
        Ok(())
    }
    
    /// 添加目录的文件指纹（递归）
    pub fn add_directory(
        &mut self,
        dir: &Path,
        pattern: Option<&str>,
        max_files: usize,
        include_hash: bool,
    ) -> Result<(), String> {
        let mut count = 0;
        
        let entries = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read dir {:?}: {}", dir, e))?;
        
        for entry in entries {
            if count >= max_files {
                break;
            }
            
            let entry = entry.map_err(|e| format!("Failed to read entry: {}", e))?;
            let path = entry.path();
            
            // 检查文件类型
            let file_type = entry.file_type()
                .map_err(|e| format!("Failed to get file type: {}", e))?;
            
            if file_type.is_file() {
                // 检查文件扩展名匹配
                if let Some(pat) = pattern {
                    if !path.extension()
                        .map(|e| e.to_string_lossy() == pat)
                        .unwrap_or(false)
                    {
                        continue;
                    }
                }
                
                self.add_file(&path, include_hash)?;
                count += 1;
            } else if file_type.is_dir() {
                // 递归处理子目录
                self.add_directory(&path, pattern, max_files - count, include_hash)?;
            }
        }
        
        Ok(())
    }
    
    /// 检测 Git 提交
    pub fn detect_git_commit(&mut self, repo_path: &Path) {
        self.git_commit = self::get_git_commit(repo_path).ok();
    }
    
    /// 验证当前文件状态是否与指纹匹配
    pub fn verify(&self) -> FingerprintVerificationResult {
        let mut changed_files = Vec::new();
        let mut missing_files = Vec::new();
        let mut unchanged_count = 0;
        
        for file in &self.files {
            let path = Path::new(&file.path);
            
            if !path.exists() {
                missing_files.push(file.path.clone());
                continue;
            }
            
            match fs::metadata(path) {
                Ok(metadata) => {
                    let current_mtime = metadata.modified()
                        .map(|t| t.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs())
                        .unwrap_or(0);
                    
                    // 检查 mtime 或 size 是否变化
                    if current_mtime != file.mtime || metadata.len() != file.size {
                        changed_files.push(file.path.clone());
                    } else {
                        // mtime 和 size 相同，检查 hash（如果有）
                        if let Some(ref stored_hash) = file.hash {
                            if let Ok(current_hash) = self::compute_file_hash(path) {
                                if &current_hash != stored_hash {
                                    changed_files.push(file.path.clone());
                                } else {
                                    unchanged_count += 1;
                                }
                            } else {
                                unchanged_count += 1; // hash 计算失败，假设未变
                            }
                        } else {
                            unchanged_count += 1;
                        }
                    }
                }
                Err(_) => {
                    missing_files.push(file.path.clone());
                }
            }
        }
        
        FingerprintVerificationResult {
            is_valid: changed_files.is_empty() && missing_files.is_empty(),
            changed_files,
            missing_files,
            unchanged_count,
            total_files: self.files.len(),
        }
    }
    
    /// 检查 Git 提交是否变化
    pub fn verify_git_commit(&self, repo_path: &Path) -> bool {
        match self::get_git_commit(repo_path) {
            Ok(current_commit) => {
                match &self.git_commit {
                    Some(stored) => stored == &current_commit,
                    None => false, // 有当前提交但没有存储的提交，认为变化了
                }
            }
            Err(_) => true, // 无法获取 Git 信息时，假设未变
        }
    }
}

/// 指纹验证结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintVerificationResult {
    /// 是否有效（无变化）
    pub is_valid: bool,
    /// 已变化的文件列表
    pub changed_files: Vec<String>,
    /// 缺失的文件列表
    pub missing_files: Vec<String>,
    /// 未变化的文件数量
    pub unchanged_count: usize,
    /// 总文件数
    pub total_files: usize,
}

/// 计算文件 SHA256 hash
pub fn compute_file_hash(path: &Path) -> Result<String, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open file {:?}: {}", path, e))?;
    
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    
    loop {
        let bytes_read = reader.read(&mut buffer)
            .map_err(|e| format!("Failed to read file: {}", e))?;
        
        if bytes_read == 0 {
            break;
        }
        
        hasher.update(&buffer[..bytes_read]);
    }
    
    let result = hasher.finalize();
    Ok(format!("{:x}", result))
}

/// 获取 Git 当前提交 hash
pub fn get_git_commit(repo_path: &Path) -> Result<String, String> {
    use std::process::Command;
    
    let output = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| format!("Failed to run git: {}", e))?;
    
    if output.status.success() {
        let commit = String::from_utf8_lossy(&output.stdout)
            .trim()
            .to_string();
        Ok(commit)
    } else {
        Err("Not a git repository or git command failed".to_string())
    }
}

/// 获取 Git 状态（是否有未提交的更改）
pub fn get_git_status(repo_path: &Path) -> Result<GitStatus, String> {
    use std::process::Command;
    
    // 检查工作树状态
    let output = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(repo_path)
        .output()
        .map_err(|e| format!("Failed to run git status: {}", e))?;
    
    let status_output = String::from_utf8_lossy(&output.stdout);
    let has_uncommitted_changes = !status_output.trim().is_empty();
    
    // 检查是否有未推送的提交
    let output = Command::new("git")
        .args(["log", "@{u}..HEAD"])
        .current_dir(repo_path)
        .output();
    
    let has_unpushed_commits = output
        .map(|o| !String::from_utf8_lossy(&o.stdout).trim().is_empty())
        .unwrap_or(false);
    
    Ok(GitStatus {
        has_uncommitted_changes,
        has_unpushed_commits,
        uncommitted_count: status_output.lines().count(),
    })
}

/// Git 状态信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitStatus {
    pub has_uncommitted_changes: bool,
    pub has_unpushed_commits: bool,
    pub uncommitted_count: usize,
}

/// 为特定主题生成指纹
pub fn create_fingerprint_for_topic(
    topic: &str,
    context: &HashMap<String, String>,
) -> Result<KnowledgeFingerprint, String> {
    let mut fingerprint = KnowledgeFingerprint::new(context);
    
    // 根据主题决定要跟踪哪些文件
    match topic {
        "project_structure" | "project_info" => {
            // 跟踪项目根目录的结构
            if let Some(project_path) = context.get("project_path") {
                let path = Path::new(project_path);
                
                // 添加关键目录
                for dir in &["src", "Cargo.toml", "package.json", "README.md"] {
                    let full_path = path.join(dir);
                    if full_path.exists() {
                        if full_path.is_dir() {
                            fingerprint.add_directory(&full_path, Some("rs"), 50, false)?;
                        } else {
                            fingerprint.add_file(&full_path, true)?;
                        }
                    }
                }
                
                // 检测 Git 提交
                fingerprint.detect_git_commit(path);
            }
        }
        "code_content" => {
            // 跟踪特定文件的内容
            if let Some(file_path) = context.get("file_path") {
                fingerprint.add_file(Path::new(file_path), true)?;
            }
        }
        "project_config" => {
            // 跟踪配置文件
            if let Some(project_path) = context.get("project_path") {
                let path = Path::new(project_path);
                for config_file in &["Cargo.toml", "package.json", "config.json", ".env"] {
                    let full_path = path.join(config_file);
                    if full_path.exists() {
                        fingerprint.add_file(&full_path, true)?;
                    }
                }
            }
        }
        _ => {
            // 其他主题，不跟踪文件变化
        }
    }
    
    Ok(fingerprint)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_file_hash() {
        // 创建一个临时文件测试
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_hash.txt");
        
        std::fs::write(&test_file, "test content").unwrap();
        
        let hash1 = compute_file_hash(&test_file).unwrap();
        let hash2 = compute_file_hash(&test_file).unwrap();
        
        assert_eq!(hash1, hash2);
        
        // 修改文件内容
        std::fs::write(&test_file, "different content").unwrap();
        let hash3 = compute_file_hash(&test_file).unwrap();
        
        assert_ne!(hash1, hash3);
        
        // 清理
        std::fs::remove_file(&test_file).ok();
    }
    
    #[test]
    fn test_fingerprint_verification() {
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join("test_fingerprint.txt");
        
        std::fs::write(&test_file, "initial content").unwrap();
        
        let mut fingerprint = KnowledgeFingerprint::new(&HashMap::new());
        fingerprint.add_file(&test_file, true).unwrap();
        
        // 初始验证应该通过
        let result = fingerprint.verify();
        assert!(result.is_valid);
        
        // 修改文件后验证应该失败
        std::fs::write(&test_file, "modified content").unwrap();
        let result = fingerprint.verify();
        assert!(!result.is_valid);
        assert!(!result.changed_files.is_empty());
        
        // 清理
        std::fs::remove_file(&test_file).ok();
    }
}
