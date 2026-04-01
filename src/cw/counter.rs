//! 计数器（Counter）实现
//!
//! 用于统计元素出现频率的数据结构，类似于 Python 的 `collections.Counter`。

use std::hash::Hash;

use crate::common::types::FastMap;

/// 计数器（Counter）
///
/// 用于统计元素出现频率的数据结构。内部使用 `FastMap`（基于 FxHasher 的 HashMap）实现，
/// 提供高性能的计数操作。
///
/// # 类型参数
///
/// * `K` - 计数的元素类型，必须实现 `Eq` 和 `Hash`
///
/// # 示例
///
/// ```rust
/// use rust_tools::cw::counter::Counter;
///
/// let mut counter: Counter<char> = Counter::new();
///
/// // 统计字符频率
/// for c in "hello world".chars() {
///     counter.inc(c);
/// }
///
/// assert_eq!(counter.get(&'l'), 3);
/// assert_eq!(counter.get(&'o'), 2);
/// assert_eq!(counter.get(&'h'), 1);
/// assert_eq!(counter.get(&'x'), 0); // 不存在的元素返回 0
///
/// // 获取最常见的元素
/// let top3 = counter.most_common(3);
/// assert_eq!(top3[0], ('l', 3));
/// assert_eq!(top3[1], ('o', 2));
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：
///   - `inc`/`add`/`dec`: O(1) 平均
///   - `get`/`contains`: O(1) 平均
///   - `most_common`: O(n log n)，n 为不同元素的数量
/// - 空间复杂度：O(n)，n 为不同元素的数量
///
/// # 注意事项
///
/// - 计数不会低于 0
/// - 当计数减到 0 时，元素会自动从计数器中移除
/// - 使用饱和加法/减法，避免溢出
#[derive(Clone, Debug, Default)]
pub struct Counter<K>
where
    K: Eq + Hash,
{
    /// 存储元素及其计数的映射
    data: FastMap<K, usize>,
    /// 所有元素的总计数
    total: usize,
}

impl<K> Counter<K>
where
    K: Eq + Hash,
{
    /// 创建一个新的空计数器
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let counter: Counter<i32> = Counter::new();
    /// assert!(counter.is_empty());
    /// assert_eq!(counter.total(), 0);
    /// ```
    pub fn new() -> Self {
        Self {
            data: FastMap::default(),
            total: 0,
        }
    }

    /// 清空计数器
    ///
    /// 移除所有元素并重置总计数。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.inc("b");
    /// counter.clear();
    /// assert!(counter.is_empty());
    /// assert_eq!(counter.total(), 0);
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
        self.total = 0;
    }

    /// 返回不同元素的数量
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.inc("b");
    /// counter.inc("a");
    /// assert_eq!(counter.len(), 2); // "a" 和 "b" 两个不同元素
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// 检查计数器是否为空
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert!(counter.is_empty());
    /// counter.inc("a");
    /// assert!(!counter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 返回所有元素的总计数
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.add("b", 3);
    /// assert_eq!(counter.total(), 4);
    /// ```
    pub fn total(&self) -> usize {
        self.total
    }

    /// 获取元素的计数值
    ///
    /// 如果元素不存在，返回 0。
    ///
    /// # 参数
    ///
    /// * `key` - 要查询的元素
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// assert_eq!(counter.get(&"a"), 1);
    /// assert_eq!(counter.get(&"b"), 0); // 不存在的元素返回 0
    /// ```
    pub fn get(&self, key: &K) -> usize {
        self.data.get(key).copied().unwrap_or(0)
    }

    /// 增加元素的计数值
    ///
    /// # 参数
    ///
    /// * `key` - 要增加计数的元素
    /// * `n` - 增加的数量
    ///
    /// # 返回值
    ///
    /// 返回增加后的新计数值
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert_eq!(counter.add("a", 5), 5);
    /// assert_eq!(counter.add("a", 3), 8);
    /// assert_eq!(counter.get(&"a"), 8);
    /// ```
    ///
    /// # 注意事项
    ///
    /// - 如果 `n` 为 0，不执行任何操作，返回当前计数值
    /// - 使用饱和加法，避免溢出
    pub fn add(&mut self, key: K, n: usize) -> usize {
        if n == 0 {
            return self.get(&key);
        }
        self.total = self.total.saturating_add(n);
        let x = self.data.entry(key).or_insert(0);
        *x = x.saturating_add(n);
        *x
    }

    /// 将元素的计数值加 1
    ///
    /// # 参数
    ///
    /// * `key` - 要增加计数的元素
    ///
    /// # 返回值
    ///
    /// 返回增加后的新计数值
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert_eq!(counter.inc("a"), 1);
    /// assert_eq!(counter.inc("a"), 2);
    /// ```
    pub fn inc(&mut self, key: K) -> usize {
        self.add(key, 1)
    }

    /// 将元素的计数值减 1
    ///
    /// # 参数
    ///
    /// * `key` - 要减少计数的元素
    ///
    /// # 返回值
    ///
    /// 返回减少后的新计数值
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.add("a", 3);
    /// assert_eq!(counter.dec(&"a"), 2);
    /// assert_eq!(counter.dec(&"a"), 1);
    /// assert_eq!(counter.dec(&"a"), 0);
    /// // 计数为 0 时，元素会被自动移除
    /// assert!(!counter.contains(&"a"));
    /// ```
    ///
    /// # 注意事项
    ///
    /// - 如果元素不存在或计数已为 0，返回 0
    /// - 当计数减到 0 时，元素会自动从计数器中移除
    pub fn dec(&mut self, key: &K) -> usize {
        let Some(curr) = self.data.get_mut(key) else {
            return 0;
        };
        if *curr == 0 {
            return 0;
        }
        *curr -= 1;
        self.total = self.total.saturating_sub(1);
        let left = *curr;
        if left == 0 {
            self.data.remove(key);
        }
        left
    }

    /// 移除元素并返回其计数值
    ///
    /// # 参数
    ///
    /// * `key` - 要移除的元素
    ///
    /// # 返回值
    ///
    /// - `Some(count)` - 如果元素存在，返回其计数值
    /// - `None` - 如果元素不存在
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.add("a", 5);
    /// assert_eq!(counter.remove(&"a"), Some(5));
    /// assert_eq!(counter.remove(&"a"), None);
    /// ```
    pub fn remove(&mut self, key: &K) -> Option<usize> {
        let v = self.data.remove(key)?;
        self.total = self.total.saturating_sub(v);
        Some(v)
    }

    /// 检查元素是否在计数器中
    ///
    /// # 参数
    ///
    /// * `key` - 要检查的元素
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// assert!(counter.contains(&"a"));
    /// assert!(!counter.contains(&"b"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// 返回计数器的迭代器
    ///
    /// 迭代顺序不确定。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.inc("b");
    ///
    /// let mut count = 0;
    /// for (key, value) in counter.iter() {
    ///     count += 1;
    /// }
    /// assert_eq!(count, 2);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&K, &usize)> {
        self.data.iter()
    }
}

impl<K> Counter<K>
where
    K: Eq + Hash + Clone,
{
    /// 返回最常见的 n 个元素及其计数值
    ///
    /// 按计数值降序排列。如果元素数量少于 n，返回所有元素。
    ///
    /// # 参数
    ///
    /// * `n` - 要返回的元素数量
    ///
    /// # 返回值
    ///
    /// 包含 (元素，计数值) 对的向量，按计数值降序排列
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::counter::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.add("a", 5);
    /// counter.add("b", 1);
    /// counter.add("c", 3);
    /// counter.add("d", 3);
    ///
    /// let top3 = counter.most_common(3);
    /// assert_eq!(top3[0], ("a", 5));
    /// // "c" 和 "d" 计数值相同，顺序不确定
    /// assert!(top3[1].1 == 3 && top3[2].1 == 3);
    /// ```
    ///
    /// # 性能特征
    ///
    /// - 时间复杂度：O(n log n)，n 为不同元素的数量
    /// - 空间复杂度：O(n)
    pub fn most_common(&self, n: usize) -> Vec<(K, usize)> {
        if n == 0 {
            return Vec::new();
        }
        let mut v: Vec<(K, usize)> = self.data.iter().map(|(k, &c)| (k.clone(), c)).collect();
        v.sort_unstable_by_key(|x| std::cmp::Reverse(x.1));
        if v.len() > n {
            v.truncate(n);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::Counter;

    #[test]
    fn test_counter_basic() {
        let mut c = Counter::new();
        assert!(c.is_empty());
        assert_eq!(c.total(), 0);
        assert_eq!(c.get(&"a"), 0);

        c.inc("a");
        c.add("b", 2);
        assert_eq!(c.len(), 2);
        assert_eq!(c.total(), 3);
        assert_eq!(c.get(&"a"), 1);
        assert_eq!(c.get(&"b"), 2);

        assert_eq!(c.dec(&"b"), 1);
        assert_eq!(c.total(), 2);
        assert_eq!(c.remove(&"a"), Some(1));
        assert!(!c.contains(&"a"));
        assert_eq!(c.total(), 1);
    }

    #[test]
    fn test_counter_most_common() {
        let mut c = Counter::new();
        c.add("a", 5);
        c.add("b", 1);
        c.add("c", 3);
        let top2 = c.most_common(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0], ("a", 5));
    }

    #[test]
    fn test_counter_iter() {
        let mut c = Counter::new();
        c.add("a", 1);
        c.add("b", 2);
        c.add("c", 3);
        
        let mut total = 0;
        for (_, &count) in c.iter() {
            total += count;
        }
        assert_eq!(total, 6);
    }

    #[test]
    fn test_counter_dec_to_zero() {
        let mut c = Counter::new();
        c.add("a", 1);
        assert_eq!(c.dec(&"a"), 0);
        assert!(!c.contains(&"a"));
    }
}
