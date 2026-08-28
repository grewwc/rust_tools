//! Counter implementation.
//!
//! A data structure for counting element frequencies, similar to Python's
//! `collections.Counter`.

use std::hash::Hash;

use crate::commonw::types::FastMap;

/// Counter.
///
/// A data structure for counting element frequencies. Implemented on top of
/// `FastMap` (a HashMap based on FxHasher), it provides high-performance
/// counting operations.
///
/// # Type Parameters
///
/// * `K` - The element type being counted; must implement `Eq` and `Hash`
///
/// # Examples
///
/// ```rust
/// use rust_tools::cw::Counter;
///
/// let mut counter: Counter<char> = Counter::new();
///
/// // Count character frequencies
/// for c in "hello world".chars() {
///     counter.inc(c);
/// }
///
/// assert_eq!(counter.get(&'l'), 3);
/// assert_eq!(counter.get(&'o'), 2);
/// assert_eq!(counter.get(&'h'), 1);
/// assert_eq!(counter.get(&'x'), 0); // missing elements return 0
///
/// // Get the most common elements
/// let top3 = counter.most_common(3);
/// assert_eq!(top3[0], ('l', 3));
/// assert_eq!(top3[1], ('o', 2));
/// ```
///
/// # Performance
///
/// - Time complexity:
///   - `inc`/`add`/`dec`: O(1) average
///   - `get`/`contains`: O(1) average
///   - `most_common`: O(n log n), where n is the number of distinct elements
/// - Space complexity: O(n), where n is the number of distinct elements
///
/// # Notes
///
/// - Counts never go below 0
/// - When a count reaches 0, the element is automatically removed from the counter
/// - Uses saturating addition/subtraction to avoid overflow
#[derive(Clone, Debug, Default)]
pub struct Counter<K>
where
    K: Eq + Hash,
{
    /// Map storing elements and their counts
    data: FastMap<K, usize>,
    /// Total count across all elements
    total: usize,
}

impl<K> Counter<K>
where
    K: Eq + Hash,
{
    /// Creates a new empty counter
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
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

    /// Clears the counter
    ///
    /// Removes all elements and resets the total count.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
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

    /// Returns the number of distinct elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.inc("b");
    /// counter.inc("a");
    /// assert_eq!(counter.len(), 2); // two distinct elements, "a" and "b"
    /// ```
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks whether the counter is empty
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert!(counter.is_empty());
    /// counter.inc("a");
    /// assert!(!counter.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the total count across all elements
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// counter.add("b", 3);
    /// assert_eq!(counter.total(), 4);
    /// ```
    pub fn total(&self) -> usize {
        self.total
    }

    /// Gets the count of an element
    ///
    /// Returns 0 if the element does not exist.
    ///
    /// # Arguments
    ///
    /// * `key` - The element to look up
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// assert_eq!(counter.get(&"a"), 1);
    /// assert_eq!(counter.get(&"b"), 0); // missing elements return 0
    /// ```
    pub fn get(&self, key: &K) -> usize {
        self.data.get(key).copied().unwrap_or(0)
    }

    /// Adds to an element's count
    ///
    /// # Arguments
    ///
    /// * `key` - The element to increment
    /// * `n` - The amount to add
    ///
    /// # Returns
    ///
    /// Returns the new count after the addition
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert_eq!(counter.add("a", 5), 5);
    /// assert_eq!(counter.add("a", 3), 8);
    /// assert_eq!(counter.get(&"a"), 8);
    /// ```
    ///
    /// # Notes
    ///
    /// - If `n` is 0, does nothing and returns the current count
    /// - Uses saturating addition to avoid overflow
    pub fn add(&mut self, key: K, n: usize) -> usize {
        if n == 0 {
            return self.get(&key);
        }
        self.total = self.total.saturating_add(n);
        let x = self.data.entry(key).or_insert(0);
        *x = x.saturating_add(n);
        *x
    }

    /// Increments an element's count by 1
    ///
    /// # Arguments
    ///
    /// * `key` - The element to increment
    ///
    /// # Returns
    ///
    /// Returns the new count after the increment
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// assert_eq!(counter.inc("a"), 1);
    /// assert_eq!(counter.inc("a"), 2);
    /// ```
    pub fn inc(&mut self, key: K) -> usize {
        self.add(key, 1)
    }

    /// Decrements an element's count by 1
    ///
    /// # Arguments
    ///
    /// * `key` - The element to decrement
    ///
    /// # Returns
    ///
    /// Returns the new count after the decrement
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.add("a", 3);
    /// assert_eq!(counter.dec(&"a"), 2);
    /// assert_eq!(counter.dec(&"a"), 1);
    /// assert_eq!(counter.dec(&"a"), 0);
    /// // When the count reaches 0, the element is removed automatically
    /// assert!(!counter.contains(&"a"));
    /// ```
    ///
    /// # Notes
    ///
    /// - Returns 0 if the element does not exist or its count is already 0
    /// - When a count reaches 0, the element is automatically removed from the counter
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

    /// Removes an element and returns its count
    ///
    /// # Arguments
    ///
    /// * `key` - The element to remove
    ///
    /// # Returns
    ///
    /// - `Some(count)` - the element's count, if the element exists
    /// - `None` - if the element does not exist
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
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

    /// Checks whether an element is in the counter
    ///
    /// # Arguments
    ///
    /// * `key` - The element to check
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.inc("a");
    /// assert!(counter.contains(&"a"));
    /// assert!(!counter.contains(&"b"));
    /// ```
    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Returns an iterator over the counter
    ///
    /// The iteration order is unspecified.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
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
    /// Returns the n most common elements with their counts
    ///
    /// Sorted by count in descending order. If there are fewer than n elements,
    /// all elements are returned.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of elements to return
    ///
    /// # Returns
    ///
    /// A vector of (element, count) pairs, sorted by count in descending order
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rust_tools::cw::Counter;
    ///
    /// let mut counter = Counter::new();
    /// counter.add("a", 5);
    /// counter.add("b", 1);
    /// counter.add("c", 3);
    /// counter.add("d", 3);
    ///
    /// let top3 = counter.most_common(3);
    /// assert_eq!(top3[0], ("a", 5));
    /// // "c" and "d" have equal counts; the order is unspecified
    /// assert!(top3[1].1 == 3 && top3[2].1 == 3);
    /// ```
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n log n), where n is the number of distinct elements
    /// - Space complexity: O(n)
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
