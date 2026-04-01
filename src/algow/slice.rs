//! 切片算法工具
//!
//! 提供在有序切片上进行二分查找的算法。

/// 二分查找左边界（第一个大于等于目标值的位置）
///
/// 在有序切片中查找第一个大于等于 `target` 的元素位置。
/// 如果所有元素都小于 `target`，返回切片长度。
///
/// # 类型参数
///
/// * `T` - 元素类型，必须实现 `PartialOrd`
///
/// # 参数
///
/// * `arr` - 有序切片（升序）
/// * `target` - 要查找的目标值
///
/// # 返回值
///
/// 返回第一个大于等于 `target` 的元素索引，如果不存在则返回切片长度。
///
/// # 示例
///
/// ```rust
/// use rust_tools::algow::bisect_left;
///
/// let arr = [1, 3, 5, 7, 9];
///
/// // 找到等于目标值的位置
/// assert_eq!(bisect_left(&arr, &5), 2);
///
/// // 找到大于目标值的第一个位置
/// assert_eq!(bisect_left(&arr, &6), 3);
///
/// // 目标值小于所有元素
/// assert_eq!(bisect_left(&arr, &0), 0);
///
/// // 目标值大于所有元素
/// assert_eq!(bisect_left(&arr, &10), 5);
///
/// // 处理重复元素，返回第一个出现的位置
/// let arr_with_dups = [1, 3, 3, 3, 5];
/// assert_eq!(bisect_left(&arr_with_dups, &3), 1);
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：O(log n)
/// - 空间复杂度：O(1)
///
/// # 注意事项
///
/// - 输入切片必须是升序排列的，否则结果未定义
/// - 对于空切片，始终返回 0
///
/// # 参见
///
/// - [`bisect_right`] - 查找右边界（第一个大于目标值的位置）
pub fn bisect_left<T: PartialOrd>(arr: &[T], target: &T) -> usize {
    if arr.is_empty() {
        return 0;
    }
    let (mut lo, mut hi) = (0_usize, arr.len());
    while lo < hi {
        let mid = (hi - lo) / 2 + lo;
        if target > &arr[mid] {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// 二分查找右边界（第一个大于目标值的位置）
///
/// 在有序切片中查找第一个大于 `target` 的元素位置。
/// 如果所有元素都小于等于 `target`，返回切片长度。
///
/// # 类型参数
///
/// * `T` - 元素类型，必须实现 `PartialOrd`
///
/// # 参数
///
/// * `arr` - 有序切片（升序）
/// * `target` - 要查找的目标值
///
/// # 返回值
///
/// 返回第一个大于 `target` 的元素索引，如果不存在则返回切片长度。
///
/// # 示例
///
/// ```rust
/// use rust_tools::algow::bisect_right;
///
/// let arr = [1, 3, 5, 7, 9];
///
/// // 找到大于目标值的第一个位置
/// assert_eq!(bisect_right(&arr, &5), 3);
///
/// // 目标值等于某个元素，返回其后一个位置
/// assert_eq!(bisect_right(&arr, &1), 1);
///
/// // 目标值小于所有元素
/// assert_eq!(bisect_right(&arr, &0), 0);
///
/// // 目标值大于所有元素
/// assert_eq!(bisect_right(&arr, &10), 5);
///
/// // 处理重复元素，返回最后一个出现位置的后一个位置
/// let arr_with_dups = [1, 3, 3, 3, 5];
/// assert_eq!(bisect_right(&arr_with_dups, &3), 4);
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：O(log n)
/// - 空间复杂度：O(1)
///
/// # 注意事项
///
/// - 输入切片必须是升序排列的，否则结果未定义
/// - 对于空切片，始终返回 0
///
/// # 参见
///
/// - [`bisect_left`] - 查找左边界（第一个大于等于目标值的位置）
pub fn bisect_right<T: PartialOrd>(arr: &[T], target: &T) -> usize {
    if arr.is_empty() {
        return 0;
    }

    let (mut lo, mut hi) = (0_usize, arr.len());
    while lo < hi {
        let mid = (hi - lo) / 2 + lo;
        if target < &arr[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

#[cfg(test)]
mod tests {
    use super::{bisect_left, bisect_right};

    #[test]
    fn test_bisect_left_basic() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(bisect_left(&arr, &5), 2);
        assert_eq!(bisect_left(&arr, &6), 3);
        assert_eq!(bisect_left(&arr, &0), 0);
        assert_eq!(bisect_left(&arr, &10), 5);
    }

    #[test]
    fn test_bisect_left_with_duplicates() {
        let arr = [1, 3, 3, 3, 5];
        assert_eq!(bisect_left(&arr, &3), 1);
    }

    #[test]
    fn test_bisect_left_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(bisect_left(&arr, &5), 0);
    }

    #[test]
    fn test_bisect_right_basic() {
        let arr = [1, 3, 5, 7, 9];
        assert_eq!(bisect_right(&arr, &5), 3);
        assert_eq!(bisect_right(&arr, &6), 3);
        assert_eq!(bisect_right(&arr, &0), 0);
        assert_eq!(bisect_right(&arr, &10), 5);
    }

    #[test]
    fn test_bisect_right_with_duplicates() {
        let arr = [1, 3, 3, 3, 5];
        assert_eq!(bisect_right(&arr, &3), 4);
    }

    #[test]
    fn test_bisect_right_empty() {
        let arr: [i32; 0] = [];
        assert_eq!(bisect_right(&arr, &5), 0);
    }

    #[test]
    fn test_bisect_with_strings() {
        let arr = ["apple", "banana", "cherry", "date"];
        assert_eq!(bisect_left(&arr, &"cherry"), 2);
        assert_eq!(bisect_right(&arr, &"cherry"), 3);
    }
}
