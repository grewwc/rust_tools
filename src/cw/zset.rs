use std::cmp::Ordering;
use std::hash::Hash;

use crate::commonw::types::FastMap;
use crate::cw::SkipSet;

#[derive(Clone, Debug, PartialEq)]
pub struct ZSetEntry<'a, K> {
    key: &'a K,
    score: f64,
}

impl<K> ZSetEntry<'_, K> {
    pub fn key(&self) -> &K {
        self.key
    }

    pub fn score(&self) -> f64 {
        self.score
    }
}

#[derive(Clone, Debug)]
struct ZSetNode<K> {
    key: K,
    score: f64,
    order: u64,
}

impl<K> ZSetNode<K> {
    fn new(key: K, score: f64, order: u64) -> Self {
        Self { key, score, order }
    }
}

impl<K> PartialEq for ZSetNode<K> {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl<K> Eq for ZSetNode<K> {}

impl<K> PartialOrd for ZSetNode<K> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<K> Ord for ZSetNode<K> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.order.cmp(&other.order))
    }
}

/// Ordered set with scores (similar to Redis ZSET).
///
/// Elements are sorted by score. K only needs `Eq + Hash`.
pub struct ZSet<K>
where
    K: Eq + Hash,
{
    tree: SkipSet<ZSetNode<K>>,
    map: FastMap<K, (f64, u64)>,
    next_order: u64,
}

impl<K> ZSet<K>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            tree: SkipSet::new(12),
            map: FastMap::default(),
            next_order: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn search_range(&self, score_low: f64, score_high: f64) -> Vec<ZSetEntry<'_, K>> {
        if score_low > score_high {
            return Vec::new();
        }
        self.tree
            .iter()
            .filter(|entry| entry.score >= score_low && entry.score <= score_high)
            .map(|entry| ZSetEntry {
                key: &entry.key,
                score: entry.score,
            })
            .collect()
    }

    /// Returns -1 if key does not exist. Otherwise returns 1-based rank.
    pub fn rank(&self, key: &K) -> isize {
        if !self.map.contains_key(key) {
            return -1;
        }
        for (idx, node) in self.tree.iter().enumerate() {
            if node.key == *key {
                return idx as isize + 1;
            }
        }
        -1
    }

    pub fn min(&self) -> Option<ZSetEntry<'_, K>> {
        self.tree.first().map(|entry| ZSetEntry {
            key: &entry.key,
            score: entry.score,
        })
    }

    pub fn max(&self) -> Option<ZSetEntry<'_, K>> {
        self.tree.last().map(|entry| ZSetEntry {
            key: &entry.key,
            score: entry.score,
        })
    }

    pub fn score(&self, key: &K) -> Option<f64> {
        self.map.get(key).map(|(s, _)| *s)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    pub fn iter(&self) -> Vec<ZSetEntry<'_, K>> {
        self.tree
            .iter()
            .map(|entry| ZSetEntry {
                key: &entry.key,
                score: entry.score,
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.tree.clear();
        self.map.clear();
        self.next_order = 0;
    }
}

impl<K> ZSet<K>
where
    K: Eq + Hash + Clone,
{
    pub fn add(&mut self, key: K, score: f64) -> bool {
        if self.map.contains_key(&key) {
            return false;
        }
        let order = self.next_order;
        self.next_order += 1;
        self.tree.insert(ZSetNode::new(key.clone(), score, order));
        self.map.insert(key, (score, order));
        true
    }

    pub fn update_score(&mut self, key: &K, score: f64) -> bool {
        let Some((old_score, old_order)) = self.map.get(key).copied() else {
            return false;
        };
        let old_node = ZSetNode::new(key.clone(), old_score, old_order);
        self.tree.remove(&old_node);
        let order = self.next_order;
        self.next_order += 1;
        self.tree.insert(ZSetNode::new(key.clone(), score, order));
        self.map.insert(key.clone(), (score, order));
        true
    }

    pub fn delete(&mut self, key: &K) -> bool {
        let Some((score, order)) = self.map.remove(key) else {
            return false;
        };
        self.tree.remove(&ZSetNode::new(key.clone(), score, order));
        true
    }

    pub fn remove_score(&mut self, key: &K) -> Option<f64> {
        let (score, order) = self.map.remove(key)?;
        self.tree.remove(&ZSetNode::new(key.clone(), score, order));
        Some(score)
    }

    pub fn pop_min(&mut self) -> Option<(K, f64)> {
        let node = self.tree.first()?;
        let key = node.key.clone();
        let score = node.score;
        let order = node.order;
        let node_to_remove = ZSetNode::new(key.clone(), score, order);
        if self.tree.remove(&node_to_remove) {
            self.map.remove(&key);
            Some((key, score))
        } else {
            None
        }
    }

    pub fn pop_max(&mut self) -> Option<(K, f64)> {
        let node = self.tree.last()?;
        let key = node.key.clone();
        let score = node.score;
        let order = node.order;
        let node_to_remove = ZSetNode::new(key.clone(), score, order);
        if self.tree.remove(&node_to_remove) {
            self.map.remove(&key);
            Some((key, score))
        } else {
            None
        }
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
}

impl<K> Default for ZSet<K>
where
    K: Eq + Hash,
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
        assert_eq!(&z.pop_min().unwrap().0, &"b");
        assert_eq!(&z.pop_max().unwrap().0, &"c");
        assert!(z.is_empty());
    }

    #[test]
    fn test_zset_with_non_ord_key() {
        use std::hash::Hash as StdHash;

        #[derive(Debug, Clone)]
        struct NotOrd {
            id: u32,
        }

        impl PartialEq for NotOrd {
            fn eq(&self, other: &Self) -> bool {
                self.id == other.id
            }
        }

        impl Eq for NotOrd {}

        impl StdHash for NotOrd {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.id.hash(state);
            }
        }

        let mut z = ZSet::new();
        let k1 = NotOrd { id: 1 };
        let k2 = NotOrd { id: 2 };
        let k3 = NotOrd { id: 3 };

        assert!(z.add(k1, 3.0));
        assert!(z.add(k2, 1.0));
        assert!(z.add(k3, 2.0));

        assert_eq!(z.rank(&NotOrd { id: 2 }), 1);
        assert_eq!(z.rank(&NotOrd { id: 3 }), 2);
        assert_eq!(z.rank(&NotOrd { id: 1 }), 3);

        let min = z.pop_min().unwrap();
        assert_eq!(min.0.id, 2);
        assert_eq!(min.1, 1.0);
    }
}
