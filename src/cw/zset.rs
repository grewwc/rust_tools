use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::hash::Hash;

use crate::common::types::FastMap;

#[derive(Clone, Debug, PartialEq)]
pub struct ZSetEntry<K>
where
    K: Clone,
{
    key: K,
    score: f64,
}

impl<K> ZSetEntry<K>
where
    K: Clone,
{
    pub fn new(key: K, score: f64) -> Self {
        Self { key, score }
    }

    pub fn key(&self) -> &K {
        &self.key
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug)]
struct ZSetNode<K>
where
    K: Ord + Clone,
{
    key: K,
    score: f64,
}

impl<K> ZSetNode<K>
where
    K: Ord + Clone,
{
    fn new(key: K, score: f64) -> Self {
        Self { key, score }
    }
}

impl<K> PartialEq for ZSetNode<K>
where
    K: Ord + Clone,
{
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<K> Eq for ZSetNode<K> where K: Ord + Clone {}

impl<K> PartialOrd for ZSetNode<K>
where
    K: Ord + Clone,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for ZSetNode<K>
where
    K: Ord + Clone,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.key.cmp(&other.key))
    }
}

pub struct ZSet<K>
where
    K: Eq + Hash + Ord + Clone,
{
    tree: BTreeSet<ZSetNode<K>>,
    map: FastMap<K, f64>,
}

impl<K> ZSet<K>
where
    K: Eq + Hash + Ord + Clone,
{
    pub fn new() -> Self {
        Self {
            tree: BTreeSet::new(),
            map: FastMap::default(),
        }
    }

    pub fn add(&mut self, key: K, score: f64) -> bool {
        if self.map.contains_key(&key) {
            return false;
        }
        self.tree.insert(ZSetNode::new(key.clone(), score));
        self.map.insert(key, score);
        true
    }

    pub fn update_score(&mut self, key: &K, score: f64) -> bool {
        let Some(old_score) = self.map.get(key).copied() else {
            return false;
        };
        let old_node = ZSetNode::new(key.clone(), old_score);
        self.tree.remove(&old_node);
        self.tree.insert(ZSetNode::new(key.clone(), score));
        self.map.insert(key.clone(), score);
        true
    }

    pub fn delete(&mut self, key: &K) -> bool {
        let Some(score) = self.map.remove(key) else {
            return false;
        };
        self.tree.remove(&ZSetNode::new(key.clone(), score));
        true
    }

    pub fn remove_score(&mut self, key: &K) -> Option<f64> {
        let score = self.map.remove(key)?;
        self.tree.remove(&ZSetNode::new(key.clone(), score));
        Some(score)
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn search_range(&self, score_low: f64, score_high: f64) -> Vec<ZSetEntry<K>> {
        if score_low > score_high {
            return Vec::new();
        }
        self.tree
            .iter()
            .filter(|entry| entry.score >= score_low && entry.score <= score_high)
            .map(|entry| ZSetEntry::new(entry.key.clone(), entry.score))
            .collect()
    }

    /// Returns -1 if key does not exist. Otherwise returns 1-based rank.
    pub fn rank(&self, key: &K) -> isize {
        if !self.map.contains_key(key) {
            return -1;
        }
        for (idx, node) in self.tree.iter().enumerate() {
            if &node.key == key {
                return idx as isize + 1;
            }
        }
        -1
    }

    pub fn min(&self) -> Option<ZSetEntry<K>> {
        self.tree
            .first()
            .map(|entry| ZSetEntry::new(entry.key.clone(), entry.score))
    }

    pub fn max(&self) -> Option<ZSetEntry<K>> {
        self.tree
            .last()
            .map(|entry| ZSetEntry::new(entry.key.clone(), entry.score))
    }

    pub fn pop_min(&mut self) -> Option<ZSetEntry<K>> {
        let node = self.tree.pop_first()?;
        self.map.remove(&node.key);
        Some(ZSetEntry::new(node.key, node.score))
    }

    pub fn pop_max(&mut self) -> Option<ZSetEntry<K>> {
        let node = self.tree.pop_last()?;
        self.map.remove(&node.key);
        Some(ZSetEntry::new(node.key, node.score))
    }

    pub fn score(&self, key: &K) -> Option<f64> {
        self.map.get(key).copied()
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn iter(&self) -> Vec<ZSetEntry<K>> {
        self.tree
            .iter()
            .map(|entry| ZSetEntry::new(entry.key.clone(), entry.score))
            .collect()
    }

    pub fn intersect(&self, another: &ZSet<K>) -> ZSet<K> {
        let mut result = ZSet::new();
        for entry in self.tree.iter() {
            if another.contains(&entry.key) {
                let _ = result.add(entry.key.clone(), entry.score);
            }
        }
        result
    }

    pub fn union(&self, another: &ZSet<K>) -> ZSet<K> {
        let mut result = ZSet::new();
        for entry in self.tree.iter() {
            let _ = result.add(entry.key.clone(), entry.score);
        }
        for entry in another.tree.iter() {
            let _ = result.add(entry.key.clone(), entry.score);
        }
        result
    }

    pub fn subtract(&mut self, another: &ZSet<K>) {
        for entry in another.tree.iter() {
            let _ = self.delete(&entry.key);
        }
    }

    pub fn clear(&mut self) {
        self.tree.clear();
        self.map.clear();
    }
}

impl<K> Default for ZSet<K>
where
    K: Eq + Hash + Ord + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::ZSet;

    #[test]
    fn test_zset_basic_and_rank() {
        let mut z = ZSet::new();
        assert!(z.add("a", 3.0));
        assert!(z.add("b", 1.0));
        assert!(z.add("c", 2.0));
        assert!(!z.add("a", 4.0));

        assert_eq!(z.rank(&"b"), 1);
        assert_eq!(z.rank(&"c"), 2);
        assert_eq!(z.rank(&"x"), -1);

        assert!(z.update_score(&"a", 0.5));
        assert_eq!(z.rank(&"a"), 1);

        let range = z.search_range(0.0, 1.1);
        assert_eq!(range.len(), 2);
    }

    #[test]
    fn test_zset_union_intersect_subtract() {
        let mut z1 = ZSet::new();
        z1.add("a", 1.0);
        z1.add("b", 2.0);

        let mut z2 = ZSet::new();
        z2.add("b", 3.0);
        z2.add("c", 4.0);

        let inter = z1.intersect(&z2);
        assert_eq!(inter.len(), 1);
        assert!(inter.contains(&"b"));

        let union = z1.union(&z2);
        assert_eq!(union.len(), 3);

        z1.subtract(&z2);
        assert_eq!(z1.len(), 1);
        assert!(z1.contains(&"a"));
    }

    #[test]
    fn test_zset_pop_min_max_and_remove_score() {
        let mut z = ZSet::new();
        z.add("a", 2.0);
        z.add("b", 1.0);
        z.add("c", 3.0);
        assert_eq!(z.remove_score(&"a"), Some(2.0));
        assert!(!z.contains(&"a"));
        assert_eq!(z.pop_min().unwrap().key(), &"b");
        assert_eq!(z.pop_max().unwrap().key(), &"c");
        assert!(z.is_empty());
    }
}
