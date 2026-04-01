//! 通用类型定义
//!
//! 本模块定义项目中广泛使用的高性能集合类型。

use std::{
    collections::{HashMap, HashSet},
    hash::BuildHasherDefault,
};

use rustc_hash::FxHasher;

/// 高性能 HashMap 类型别名
///
/// 使用 `FxHasher`（来自 `rustc-hash` crate）作为哈希函数，
/// 在大多数场景下比标准的 `HashMap` 更快。
///
/// # 类型参数
///
/// * `K` - 键类型
/// * `V` - 值类型
///
/// # 示例
///
/// ```rust
/// use rust_tools::commonw::FastMap;
///
/// let mut map: FastMap<&str, i32> = FastMap::default();
/// map.insert("a", 1);
/// map.insert("b", 2);
/// map.insert("c", 3);
///
/// assert_eq!(map.get(&"a"), Some(&1));
/// assert_eq!(map.len(), 3);
/// ```
///
/// # 性能特征
///
/// - 平均时间复杂度：
///   - 插入：O(1)
///   - 查找：O(1)
///   - 删除：O(1)
/// - 空间复杂度：O(n)
///
/// # 注意事项
///
/// - `FxHasher` 不是加密安全的，不适用于需要抗哈希碰撞攻击的场景
/// - 对于大多数应用程序场景，`FxHasher` 提供更好的性能
///
/// # 参见
///
/// - [`FastSet`] - 基于 FxHasher 的 HashSet 类型别名
pub type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// 高性能 HashSet 类型别名
///
/// 使用 `FxHasher`（来自 `rustc-hash` crate）作为哈希函数，
/// 在大多数场景下比标准的 `HashSet` 更快。
///
/// # 类型参数
///
/// * `T` - 元素类型
///
/// # 示例
///
/// ```rust
/// use rust_tools::commonw::FastSet;
///
/// let mut set: FastSet<i32> = FastSet::default();
/// set.insert(1);
/// set.insert(2);
/// set.insert(3);
///
/// assert!(set.contains(&1));
/// assert!(!set.contains(&4));
/// assert_eq!(set.len(), 3);
/// ```
///
/// # 性能特征
///
/// - 平均时间复杂度：
///   - 插入：O(1)
///   - 查找：O(1)
///   - 删除：O(1)
/// - 空间复杂度：O(n)
///
/// # 注意事项
///
/// - `FxHasher` 不是加密安全的，不适用于需要抗哈希碰撞攻击的场景
/// - 对于大多数应用程序场景，`FxHasher` 提供更好的性能
///
/// # 参见
///
/// - [`FastMap`] - 基于 FxHasher 的 HashMap 类型别名
pub type FastSet<T> = HashSet<T, BuildHasherDefault<FxHasher>>;

#[cfg(test)]
mod tests {
    use super::{FastMap, FastSet};

    #[test]
    fn test_fast_map_basic() {
        let mut map: FastMap<&str, i32> = FastMap::default();
        map.insert("a", 1);
        map.insert("b", 2);
        
        assert_eq!(map.get(&"a"), Some(&1));
        assert_eq!(map.get(&"b"), Some(&2));
        assert_eq!(map.get(&"c"), None);
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_fast_set_basic() {
        let mut set: FastSet<i32> = FastSet::default();
        set.insert(1);
        set.insert(2);
        set.insert(3);
        
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&4));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn test_fast_map_from_iter() {
        let map: FastMap<&str, i32> = [("a", 1), ("b", 2), ("c", 3)].into_iter().collect();
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&"a"), Some(&1));
    }

    #[test]
    fn test_fast_set_from_iter() {
        let set: FastSet<i32> = [1, 2, 3, 3, 2, 1].into_iter().collect();
        assert_eq!(set.len(), 3);
        assert!(set.contains(&1));
    }
}
