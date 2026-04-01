//! 文本剪贴板内容处理
//!
//! 提供文本剪贴板的读取、写入和文件操作功能。
//! 支持本地剪贴板和 SSH 会话中的 OSC52 协议。

use std::{
    fmt::Display,
    fs,
    io::{self, Error, Read, Write},
    time::Duration,
};

use crate::commonw::filename::add_suffix;

/// 非文本文件错误
#[derive(Debug)]
struct NonTextErr(String);

impl NonTextErr {
