//! # Algorithm Tools
//!
//! This module provides common algorithm implementations, primarily for
//! efficient lookups over sorted data.
//!
//! ## Overview
//!
//! - [`bisect_left`] - binary search lower bound (first position >= target)
//! - [`bisect_right`] - binary search upper bound (first position > target)
//!
//! ## Usage examples
//!
//! ### Basic binary search
//!
//! ```rust
//! use rust_tools::algow::{bisect_left, bisect_right};
//!
//! let arr = [1, 3, 5, 7, 9];
//!
//! // Find the element position
//! let pos = bisect_left(&arr, &5);
//! assert_eq!(pos, 2);
//!
//! // Find the insertion position
//! let insert_pos = bisect_right(&arr, &6);
//! assert_eq!(insert_pos, 3);
//! ```
//!
//! ### Handling duplicate elements
//!
//! ```rust
//! use rust_tools::algow::{bisect_left, bisect_right};
//!
//! let arr = [1, 3, 3, 3, 5];
//!
//! // Lower bound: position of the first 3
//! let left = bisect_left(&arr, &3);
//! assert_eq!(left, 1);
//!
//! // Upper bound: position right after the last 3
//! let right = bisect_right(&arr, &3);
//! assert_eq!(right, 4);
//!
//! // Range of all elements equal to 3: [left, right)
//! assert_eq!(&arr[left..right], &[3, 3, 3]);
//! ```
//!
//! ### Working with string slices
//!
//! ```rust
//! use rust_tools::algow::bisect_left;
//!
//! let words = ["apple", "banana", "cherry", "date"];
//! let pos = bisect_left(&words, &"cherry");
//! assert_eq!(pos, 2);
//! ```
//!
//! ## Performance characteristics
//!
//! All algorithms run in O(log n) time and O(1) space.

pub mod slice;

// Re-export the commonly used functions
pub use slice::{bisect_left, bisect_right};
