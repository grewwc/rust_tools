//! # 剪贴板工具 (Clipboard Tools)
//!
//! 本模块提供跨平台的剪贴板操作功能，支持文本、图片和二进制数据的读写。
//!
//! ## 子模块概览
//!
//! - [`binary_content`] - 二进制剪贴板内容处理
//! - [`image_content`] - 图片剪贴板内容处理
//! - [`string_content`] - 文本剪贴板内容处理
//!
//! ## 功能特性
//!
//! ### 本地剪贴板
//!
//! 使用 `arboard` crate 提供跨平台的本地剪贴板访问：
//!
//! - 文本读写
//! - 图片读写
//! - 二进制数据读写
//!
//! ### 远程会话支持 (OSC52)
//!
//! 在 SSH 会话中，通过 OSC52 转义序列实现剪贴板操作：
//!
//! - 自动检测 SSH 会话
//! - 支持终端剪贴板查询
//! - Base64 编码传输
//!
//! ## 使用示例
//!
//! ### 文本剪贴板操作
//!
//! ```rust,no_run
//! use rust_tools::clipboardw::{get_clipboard_content, set_clipboard_content};
//!
//! // 读取剪贴板文本
//! let text = get_clipboard_content();
//! println!("剪贴板内容：{}", text);
//!
//! // 写入剪贴板文本
//! set_clipboard_content("Hello, World!").expect("设置剪贴板失败");
//! ```
//!
//! ### 文件操作
//!
//! ```rust,no_run
//! use rust_tools::clipboardw::{save_to_file, copy_from_file};
//!
//! // 保存剪贴板内容到文件
//! save_to_file("output.txt").expect("保存失败");
//!
//! // 从文件复制内容到剪贴板
//! copy_from_file("input.txt").expect("复制失败");
//! ```
//!
//! ## 注意事项
//!
//! - 在 SSH 会话中会自动切换到 OSC52 模式
//! - OSC52 需要终端支持
//! - 某些终端可能需要配置才能启用 OSC52

pub mod binary_content;
pub mod image_content;
pub mod string_content;

// 重新导出常用函数
pub use string_content::{
    copy_from_file,
    get_clipboard_content,
    get_clipboard_raw_bytes_via_osc52,
    save_to_file,
    set_clipboard_content,
};
