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
