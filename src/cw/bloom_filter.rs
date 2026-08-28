//! Bloom filter implementation.
//!
//! A bloom filter is a highly space-efficient probabilistic data structure for
//! testing whether an element is in a set. It may produce false positives, but
//! never produces false negatives.

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

/// A bloom filter.
///
/// A bloom filter is a highly space-efficient probabilistic data structure for
/// testing whether an element is in a set.
///
/// ## Features
///
/// - **May produce false positives**: if `contains` returns `true`, the element
///   may be in the set (not guaranteed)
/// - **Never produces false negatives**: if `contains` returns `false`, the
///   element is definitely not in the set
/// - **Space-efficient**: uses less memory than structures like a plain HashSet
///
/// ## How it works
///
/// A bloom filter uses a bit array and several hash functions. When inserting
/// an element, the hash functions compute multiple positions and set those
/// bits to 1. When querying an element, the same positions are computed; if
/// all of them are 1, the element may be present.
///
/// ## Construction
///
/// There are two ways to create a bloom filter:
///
/// 1. [`BloomFilter::new`] - specify the bit count and hash function count directly
/// 2. [`BloomFilter::with_rate`] - specify the expected item count and
///   false-positive rate; optimal parameters are computed automatically
///
/// # Examples
///
/// ```rust
/// use rust_tools::cw::BloomFilter;
///
/// // Approach 1: specify parameters directly
/// let mut bf = BloomFilter::new(1000, 3); // 1000 bits, 3 hash functions
/// bf.insert(&"hello".to_string());
/// bf.insert(&"world".to_string());
/// assert!(bf.contains(&"hello".to_string()));
/// assert!(bf.contains(&"world".to_string()));
///
/// // Approach 2: compute automatically from the expected item count and
/// // false-positive rate
/// let mut bf2 = BloomFilter::with_rate(10000, 0.01); // 10000 expected items, 1% false-positive rate
/// bf2.insert(&"rust".to_string());
/// assert!(bf2.contains(&"rust".to_string()));
/// ```
///
/// # Performance characteristics
///
/// - Time complexity:
///   - `insert`: O(k), where k is the number of hash functions
///   - `contains`: O(k)
/// - Space complexity: O(m), where m is the number of bits
///
/// # Caveats
///
/// - Bloom filters do not support deletion (unless using a counting bloom filter)
/// - The false-positive rate grows as more elements are inserted
/// - Choosing good parameters matters: more bits and a well-suited hash
///   function count lower the false-positive rate
pub struct BloomFilter {
    /// Bit array; each u64 stores 64 bits
    bits: Vec<u64>,
    /// Total number of bits
    bit_count: usize,
    /// Number of hash functions
    hash_count: u32,
}

impl BloomFilter {
    /// Creates a new bloom filter
    ///
    /// # Parameters
    ///
    /// * `bit_count` - size of the bit array (number of bits)
    /// * `hash_count` - number of hash functions to use
    ///
    /// # Examples
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

    /// Creates a bloom filter from the expected item count and false-positive rate
    ///
    /// This method automatically computes the optimal bit count and hash
    /// function count.
    ///
    /// # Parameters
    ///
    /// * `expected_items` - expected number of inserted elements
    /// * `false_positive_rate` - desired false-positive rate (between 0.0 and 1.0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// // 10000 expected items, 1% false-positive rate
    /// let bf = BloomFilter::with_rate(10000, 0.01);
    /// println!("位数：{}, 哈希函数数量：{}", bf.bit_count(), bf.hash_count());
    /// ```
    ///
    /// # Formulas
    ///
    /// - Optimal bit count m = -(n * ln(p)) / (ln(2)^2)
    /// - Optimal hash function count k = (m/n) * ln(2)
    ///
    /// where n is the element count and p the false-positive rate
    pub fn with_rate(expected_items: usize, false_positive_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = false_positive_rate.clamp(1e-12, 0.999_999_999_999);
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil().max(1.0) as usize;
        let k = ((m as f64 / n) * ln2).round().max(1.0) as u32;
        Self::new(m, k)
    }

    /// Clears the bloom filter
    ///
    /// Resets all bits to 0.
    ///
    /// # Examples
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

    /// Returns the number of bits in the bloom filter
    ///
    /// # Examples
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

    /// Returns the number of hash functions
    ///
    /// # Examples
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

    /// Inserts an element into the bloom filter
    ///
    /// # Type parameters
    ///
    /// * `T` - a hashable type
    ///
    /// # Examples
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

    /// Checks whether the element may be present in the bloom filter
    ///
    /// # Returns
    ///
    /// - `true` - the element **may** be in the set (possibly a false positive)
    /// - `false` - the element is **definitely** not in the set
    ///
    /// # Type parameters
    ///
    /// * `T` - a hashable type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::BloomFilter;
    ///
    /// let mut bf = BloomFilter::new(1000, 3);
    /// bf.insert(&"hello".to_string());
    ///
    /// assert!(bf.contains(&"hello".to_string())); // guaranteed true
    /// assert!(!bf.contains(&"world".to_string())); // may be false or true (false positive)
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

    /// Computes the bit index for the i-th hash function
    ///
    /// Uses double hashing: h(i) = h1 + i * h2
    fn index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let mixed = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (mixed % (self.bit_count as u64)) as usize
    }

    /// Sets the given bit to 1
    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Gets the value of the given bit
    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

    /// Computes two hash values for double hashing
    ///
    /// Generates two independent hash values with two different hash seeds,
    /// then derives multiple hash values by linear combination.
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
    /// Creates the default bloom filter
    ///
    /// Default configuration: 1024 expected items, 1% false-positive rate
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
