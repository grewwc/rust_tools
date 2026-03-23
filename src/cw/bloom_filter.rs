use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

pub struct BloomFilter {
    bits: Vec<u64>,
    bit_count: usize,
    hash_count: u32,
}

impl BloomFilter {
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

    pub fn with_rate(expected_items: usize, false_positive_rate: f64) -> Self {
        let n = expected_items.max(1) as f64;
        let p = false_positive_rate.clamp(1e-12, 0.999_999_999_999);
        let ln2 = std::f64::consts::LN_2;
        let m = (-(n * p.ln()) / (ln2 * ln2)).ceil().max(1.0) as usize;
        let k = ((m as f64 / n) * ln2).round().max(1.0) as u32;
        Self::new(m, k)
    }

    pub fn clear(&mut self) {
        self.bits.fill(0);
    }

    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    pub fn hash_count(&self) -> u32 {
        self.hash_count
    }

    pub fn insert<T: Hash>(&mut self, item: &T) {
        let (h1, h2) = self.hash_pair(item);
        for i in 0..self.hash_count {
            let idx = self.index(h1, h2, i);
            self.set_bit(idx);
        }
    }

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

    fn index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let mixed = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (mixed % (self.bit_count as u64)) as usize
    }

    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] & (1u64 << bit)) != 0
    }

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
}
