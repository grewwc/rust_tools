//! 布隆过滤器（Bloom Filter）实现
//!
//! 布隆过滤器是一种空间效率很高的概率型数据结构，用于测试一个元素是否在一个集合中。
//! 它可能会产生假阳性（false positive），但不会产生假阴性（false negative）。

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

/// 布隆过滤器（Bloom Filter）
///
/// 布隆过滤器是一种空间效率很高的概率型数据结构，用于测试一个元素是否在一个集合中。
///
/// ## 特性
///
/// - **可能产生假阳性**：如果 `contains` 返回 `true`，元素可能存在集合中（但不一定）
/// - **不会产生假阴性**：如果 `contains` 返回 `false`，元素一定不在集合中
/// - **空间效率高**：比传统的 HashSet 等结构占用更少的内存
///
/// ## 工作原理
///
/// 布隆过滤器使用一个位数组和多个哈希函数。当插入一个元素时，使用多个哈希函数
/// 计算出多个位置，并将这些位置的位设置为 1。当查询一个元素时，同样计算这些位置，
/// 如果所有位置都是 1，则认为元素可能存在。
///
/// ## 创建方式
///
/// 有两种创建布隆过滤器的方式：
///
/// 1. [`BloomFilter::new`] - 直接指定位数和哈希函数数量
/// 2. [`BloomFilter::with_rate`] - 指定期望元素数量和假阳性率，自动计算最优参数
///
/// # 示例
///
/// ```rust
/// use rust_tools::cw::BloomFilter;
///
/// // 方式 1：直接指定参数
/// let mut bf = BloomFilter::new(1000, 3); // 1000 位，3 个哈希函数
/// bf.insert(&"hello".to_string());
/// bf.insert(&"world".to_string());
/// assert!(bf.contains(&"hello".to_string()));
/// assert!(bf.contains(&"world".to_string()));
///
/// // 方式 2：根据期望元素数量和假阳性率自动计算
/// let mut bf2 = BloomFilter::with_rate(10000, 0.01); // 期望 10000 个元素，1% 假阳性率
/// bf2.insert(&"rust".to_string());
/// assert!(bf2.contains(&"rust".to_string()));
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：
///   - `insert`: O(k)，k 为哈希函数数量
///   - `contains`: O(k)
/// - 空间复杂度：O(m)，m 为位数
///
/// # 注意事项
///
/// - 布隆过滤器不支持删除操作（除非使用计数布隆过滤器）
/// - 假阳性率随着插入元素数量的增加而增加
/// - 选择合适的参数很重要：位数越多、哈希函数数量越合适，假阳性率越低
pub struct BloomFilter {
    /// 位数组，每 u64 存储 64 位
    bits: Vec<u64>,
    /// 总位数
    bit_count: usize,
    /// 哈希函数数量
    hash_count: u32,
}

impl BloomFilter {
    /// 创建一个新的布隆过滤器
    ///
    /// # 参数
    ///
    /// * `bit_count` - 位数组的大小（位数）
    /// * `hash_count` - 使用的哈希函数数量
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let bf = BloomFilter::new(1024, 3);
    /// assert_eq!(bf.bit_count(), 1024);
    /// assert_eq!(bf.hash_count(), 3);
    /// ```
    pub fn new(bit_count: usize, hash_count: u32) -> Self {
        let bit_count = bit_count.max(1);
        let hash_count = hash_count.max(1);
        let words = bit_count.div_ceil(64);
        Self {
            bits: vec![0; words],
            bit_count,
            hash_count,
        }
    }

    /// 根据期望的元素数量和假阳性率创建布隆过滤器
    ///
    /// 此方法会自动计算最优的位数和哈希函数数量。
    ///
    /// # 参数
    ///
    /// * `expected_items` - 期望插入的元素数量
    /// * `false_positive_rate` - 期望的假阳性率（0.0 到 1.0 之间）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// // 期望存储 10000 个元素，假阳性率 1%
    /// let bf = BloomFilter::with_rate(10000, 0.01);
    /// println!("位数：{}, 哈希函数数量：{}", bf.bit_count(), bf.hash_count());
    /// ```
    ///
    /// # 计算公式
    ///
    /// - 最优位数 m = -(n * ln(p)) / (ln(2)^2)
    /// - 最优哈希函数数量 k = (m/n) * ln(2)
    ///
    /// 其中 n 为元素数量，p 为假阳性率
    pub fn with_rate(expected_items: usize, false_positive_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = false_positive_rate.clamp(1e-12, 0.999_999_999_999);
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil().max(1.0) as usize;
        let k = ((m as f64 / n) * ln2).round().max(1.0) as u32;
        Self::new(m, k)
    }

    /// 清空布隆过滤器
    ///
    /// 将所有位重置为 0。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let mut bf = BloomFilter::new(128, 3);
    /// bf.insert(&"hello".to_string());
    /// assert!(bf.contains(&"hello".to_string()));
    /// bf.clear();
    /// assert!(!bf.contains(&"hello".to_string()));
    /// ```
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    /// 返回布隆过滤器的位数
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let bf = BloomFilter::new(1024, 3);
    /// assert_eq!(bf.bit_count(), 1024);
    /// ```
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// 返回哈希函数的数量
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let bf = BloomFilter::new(1024, 5);
    /// assert_eq!(bf.hash_count(), 5);
    /// ```
    pub fn hash_count(&self) -> u32 {
        self.hash_count
    }

    /// 向布隆过滤器中插入一个元素
    ///
    /// # 类型参数
    ///
    /// * `T` - 可哈希的类型
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let mut bf = BloomFilter::new(1000, 3);
    /// bf.insert(&"hello".to_string());
    /// bf.insert(&"world".to_string());
    /// ```
    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.hash_count {
            let idx = self.index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// 检查元素是否可能存在于布隆过滤器中
    ///
    /// # 返回值
    ///
    /// - `true` - 元素**可能**存在于集合中（可能是假阳性）
    /// - `false` - 元素**一定**不存在于集合中
    ///
    /// # 类型参数
    ///
    /// * `T` - 可哈希的类型
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let mut bf = BloomFilter::new(1000, 3);
    /// bf.insert(&"hello".to_string());
    ///
    /// assert!(bf.contains(&"hello".to_string())); // 一定为 true
    /// assert!(!bf.contains(&"world".to_string())); // 可能为 false 或 true（假阳性）
    /// ```
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.hash_count {
            let idx = self.index(h1, h2, i);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// 计算第 i 个哈希函数对应的位索引
    ///
    /// 使用双重哈希技术：h(i) = h1 + i * h2
    fn index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let mixed = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (mixed % (self.bit_count as u64)) as usize
    }

    /// 设置指定位为 1
    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// 获取指定位的值
    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    /// 计算两个哈希值用于双重哈希技术
    ///
    /// 使用两个不同的哈希种子生成两个独立的哈希值，
    /// 然后通过线性组合生成多个哈希值。
    fn hash_pair<T: Hash>(&self, item: &T) -> (u64, u64) {
        let mut a = FxHasher::default();
        item.hash(&mut a);
        0x9e37_79b9_7f4a_7c15u64.hash(&mut a);
        let h1 = a.finish();

        let mut b = std::collections::hash_map::DefaultHasher::new();
        item.hash(&mut b);
        0x243f_6a88_85a3_08d3u64.hash(&mut b);
        let mut h2 = b.finish();
        if h2 == 0 {
            h2 = 0x27d4_eb2d;
        }
        (h1, h2)
    }
}

impl Default for BloomFilter {
    /// 创建默认布隆过滤器
    ///
    /// 默认配置：期望 1024 个元素，假阳性率 1%
    fn default() -> Self {
        Self::with_rate(1024, 0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::BloomFilter;

    #[test]
    fn test_bloom_filter_no_false_negative() {
        let mut bf = BloomFilter::with_rate(100, 0.01);
        let items = ["a", "b", "c", "hello", "world", "rust_tools"];
        for x in items {
            bf.insert(&x);
        }
        for x in items {
            assert!(bf.contains(&x));
        }
    }

    #[test]
    fn test_bloom_filter_clear() {
        let mut bf = BloomFilter::new(128, 3);
        bf.insert(&"hello");
        assert!(bf.contains(&"hello"));
        bf.clear();
        assert!(!bf.contains(&"hello"));
    }

    #[test]
    fn test_bloom_filter_params_are_sane() {
        let bf = BloomFilter::with_rate(1000, 0.01);
        assert!(bf.bit_count() >= 1);
        assert!(bf.hash_count() >= 1);
    }

    #[test]
    fn test_bloom_filter_default() {
        let bf = BloomFilter::default();
        assert!(bf.bit_count() >= 1);
        assert!(bf.hash_count() >= 1);
    }
}
