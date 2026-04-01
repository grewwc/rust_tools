//! # 算法工具 (Algorithm Tools)
//!
//! 本模块提供常用的算法实现，主要用于在有序数据上进行高效查找。
//!
//! ## 功能概览
//!
//! - [`bisect_left`] - 二分查找左边界（第一个大于等于目标值的位置）
//! - [`bisect_right`] - 二分查找右边界（第一个大于目标值的位置）
//!
//! ## 使用示例
//!
//! ### 基本二分查找
//!
//! ```rust
//! use rust_tools::algow::{bisect_left, bisect_right};
//!
//! let arr = [1, 3, 5, 7, 9];
//!
//! // 查找元素位置
//! let pos = bisect_left(&arr, &5);
//! assert_eq!(pos, 2);
//!
//! // 查找插入位置
//! let insert_pos = bisect_right(&arr, &6);
//! assert_eq!(insert_pos, 3);
//! ```
//!
//! ### 处理重复元素
//!
//! ```rust
//! use rust_tools::algow::{bisect_left, bisect_right};
//!
//! let arr = [1, 3, 3, 3, 5];
//!
//! // 左边界：第一个 3 的位置
//! let left = bisect_left(&arr, &3);
//! assert_eq!(left, 1);
//!
//! // 右边界：最后一个 3 的后一个位置
//! let right = bisect_right(&arr, &3);
//! assert_eq!(right, 4);
//!
//! // 所有等于 3 的元素范围：[left, right)
//! assert_eq!(&arr[left..right], &[3, 3, 3]);
//! ```
//!
//! ### 在字符串切片上使用
//!
//! ```rust
//! use rust_tools::algow::bisect_left;
//!
//! let words = ["apple", "banana", "cherry", "date"];
//! let pos = bisect_left(&words, &"cherry");
//! assert_eq!(pos, 2);
//! ```
//!
//! ## 性能特征
//!
//! 所有算法的时间复杂度均为 O(log n)，空间复杂度为 O(1)。

pub mod slice;

// 重新导出常用函数
pub use slice::{bisect_left, bisect_right};
