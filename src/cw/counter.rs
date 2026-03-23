use std::hash::Hash;

use crate::common::types::FastMap;

#[derive(Clone, Debug, Default)]
pub struct Counter<K>
where
    K: Eq + Hash,
{
    data: FastMap<K, usize>,
    total: usize,
}

impl<K> Counter<K>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            data: FastMap::default(),
            total: 0,
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.total = 0;
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn get(&self, key: &K) -> usize {
        self.data.get(key).copied().unwrap_or(0)
    }

    pub fn add(&mut self, key: K, n: usize) -> usize {
        if n == 0 {
            return self.get(&key);
        }
        self.total = self.total.saturating_add(n);
        let x = self.data.entry(key).or_insert(0);
        *x = x.saturating_add(n);
        *x
    }

    pub fn inc(&mut self, key: K) -> usize {
        self.add(key, 1)
    }

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

    pub fn remove(&mut self, key: &K) -> Option<usize> {
        let v = self.data.remove(key)?;
        self.total = self.total.saturating_sub(v);
        Some(v)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &usize)> {
        self.data.iter()
    }
}

impl<K> Counter<K>
where
    K: Eq + Hash + Clone,
{
    pub fn most_common(&self, n: usize) -> Vec<(K, usize)> {
        if n == 0 {
            return Vec::new();
        }
        let mut v: Vec<(K, usize)> = self.data.iter().map(|(k, &c)| (k.clone(), c)).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1));
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
        assert_eq!(c.contains(&"a"), false);
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
}
