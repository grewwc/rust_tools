//! # 命令执行工具 (Command Tools)
//!
//! 本模块提供系统命令执行功能，支持超时控制、工作目录设置等特性。
//!
//! ## 功能概览
//!
//! - [`run_cmd`] - 执行命令并返回输出
//! - [`run_cmd_output`] - 执行命令并返回完整的 `Output` 对象
//! - [`run_cmd_output_with_timeout`] - 带超时控制的命令执行
//! - [`RunCmdOptions`] - 命令执行选项
//!
//! ## 使用示例
//!
//! ### 基本命令执行
//!
//! ```rust,no_run
//! use rust_tools::cmd::run_cmd;
//!
//! let output = run_cmd("echo Hello, World!").expect("命令执行失败");
//! println!("输出：{}", output);
//! ```
//!
//! ### 带工作目录执行
//!
//! ```rust,no_run
//! use rust_tools::cmd::{run_cmd_output, RunCmdOptions};
//!
//! let opts = RunCmdOptions {
//!     cwd: Some("/tmp"),
//! };
//! let output = run_cmd_output("pwd", opts).expect("命令执行失败");
//! println!("当前目录：{}", String::from_utf8_lossy(&output.stdout));
//! ```
//!
//! ### 带超时控制
//!
//! ```rust,no_run
//! use rust_tools::cmd::run_cmd_output_with_timeout;
//! use std::time::Duration;
//!
//! let output = run_cmd_output_with_timeout(
//!     "sleep 10",
//!     Default::default(),
//!     Duration::from_secs(2),
//! );
//!
//! match output {
//!     Ok(_) => println!("命令完成"),
//!     Err(e) if e.kind() == std::io::ErrorKind::TimedOut => {
//!         println!("命令超时")
//!     }
//!     Err(e) => println!("其他错误：{}", e),
//! }
//! ```
//!
//! ## 特性
//!
//! - **自动 Shell 检测**：根据命令内容自动决定是否使用 Shell
//! - **超时控制**：支持设置命令执行超时时间
//! - **工作目录**：可指定命令执行的工作目录
//! - **用户路径扩展**：自动展开 `~` 为用户主目录
//!
//! ## 安全注意事项
//!
//! - 避免直接执行包含用户输入的命令
//! - 对于复杂命令，考虑使用参数化方式
//! - 注意命令注入风险

pub mod run;

// 重新导出常用类型和函数
pub use run::{run_cmd, run_cmd_output, run_cmd_output_with_timeout, RunCmdOptions};
