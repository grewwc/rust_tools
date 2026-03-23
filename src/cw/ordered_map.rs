use std::{borrow::Borrow, fmt, hash::Hash};

use crate::common::types::FastMap;

#[derive(Clone)]
pub struct OrderedMap<K, V> {
    order: Vec<K>,
    map: FastMap<K, V>,
}

impl<K, V> OrderedMap<K, V>
where
    K: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            map: FastMap::default(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            order: Vec::with_capacity(cap),
            map: FastMap::with_capacity_and_hasher(cap, Default::default()),
        }
    }

    pub fn clear(&mut self) {
        self.order.clear();
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn capacity(&self) -> usize {
        self.order.capacity()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.order.reserve(additional);
        self.map.reserve(additional);
    }

    pub fn insert(&mut self, k: K, v: V) -> bool {
        match self.map.entry(k) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                e.insert(v);
                false
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                self.order.push(e.key().clone());
                e.insert(v);
                true
            }
        }
    }

    pub fn insert_if_absent(&mut self, k: K, v: V) -> bool {
        match self.map.entry(k) {
            std::collections::hash_map::Entry::Occupied(_) => false,
            std::collections::hash_map::Entry::Vacant(e) => {
                self.order.push(e.key().clone());
                e.insert(v);
                true
            }
        }
    }

    pub(crate) fn remove_exists<Q>(&mut self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        if self.map.remove(k).is_some() {
            self.order.retain(|x| x.borrow() != k);
            return true;
        }
        false
    }

    pub fn remove<Q>(&mut self, k: &Q)
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.remove_exists(k);
    }

    pub fn remove_value<Q>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let val = self.map.remove(k)?;
        self.order.retain(|x| x.borrow() != k);
        Some(val)
    }

    pub fn remove_entry<Q>(&mut self, k: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        let pos = self.order.iter().position(|x| x.borrow() == k)?;
        let key = self.order.get(pos)?.clone();
        let val = self.map.remove(k)?;
        self.order.remove(pos);
        Some((key, val))
    }

    pub fn pop_front(&mut self) -> Option<(K, V)> {
        if self.order.is_empty() {
            return None;
        }
        let key = self.order.remove(0);
        let val = self.map.remove(&key)?;
        Some((key, val))
    }

    pub fn pop_back(&mut self) -> Option<(K, V)> {
        let key = self.order.pop()?;
        let val = self.map.remove(&key)?;
        Some((key, val))
    }

    pub fn contains_key<Q>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.map.contains_key(k)
    }

    pub fn get<Q>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.map.get(k)
    }

    pub fn get_mut<Q>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.map.get_mut(k)
    }

    pub fn front(&self) -> Option<(&K, &V)> {
        let k = self.order.first()?;
        let v = self.map.get(k)?;
        Some((k, v))
    }

    pub fn back(&self) -> Option<(&K, &V)> {
        let k = self.order.last()?;
        let v = self.map.get(k)?;
        Some((k, v))
    }

    pub fn get_index(&self, index: usize) -> Option<(&K, &V)> {
        let k = self.order.get(index)?;
        let v = self.map.get(k)?;
        Some((k, v))
    }

    pub fn key_index<Q>(&self, k: &Q) -> Option<usize>
    where
        K: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.order.iter().position(|x| x.borrow() == k)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.order.iter()
    }

    pub fn values(&self) -> Vec<&V> {
        self.order.iter().filter_map(|k| self.map.get(k)).collect()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.order
            .iter()
            .filter_map(|k| self.map.get(k).map(|v| (k, v)))
    }
}

impl<K, V> Default for OrderedMap<K, V>
where
    K: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> fmt::Debug for OrderedMap<K, V>
where
    K: fmt::Debug + Eq + Hash,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.map.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::OrderedMap;

    #[test]
    fn test_ordered_map_order_and_pop() {
        let mut m = OrderedMap::new();
        assert!(m.insert("a", 1));
        assert!(m.insert("b", 2));
        assert!(!m.insert("a", 3));
        let keys: Vec<_> = m.keys().copied().collect();
        assert_eq!(keys, vec!["a", "b"]);
        assert_eq!(m.front(), Some((&"a", &3)));
        assert_eq!(m.back(), Some((&"b", &2)));

        assert_eq!(m.pop_front(), Some(("a", 3)));
        assert_eq!(m.pop_back(), Some(("b", 2)));
        assert!(m.is_empty());
    }

    #[test]
    fn test_ordered_map_remove_entry_and_insert_if_absent() {
        let mut m = OrderedMap::new();
        assert!(m.insert_if_absent(1, "a"));
        assert!(!m.insert_if_absent(1, "b"));
        assert_eq!(m.get(&1), Some(&"a"));

        assert_eq!(m.remove_entry(&1), Some((1, "a")));
        assert_eq!(m.remove_entry(&1), None);
        assert!(m.is_empty());
    }
}
