use std::{borrow::Borrow, fmt, hash::Hash};

use crate::cw::ordered_map::OrderedMap;

#[derive(Clone)]
pub struct OrderedSet<T> {
    map: OrderedMap<T, ()>,
}

impl<T> OrderedSet<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            map: OrderedMap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: OrderedMap::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn capacity(&self) -> usize {
        self.map.capacity()
    }

    pub fn insert(&mut self, k: T) -> bool {
        self.map.insert(k, ())
    }

    pub fn insert_if_absent(&mut self, k: T) -> bool {
        self.map.insert_if_absent(k, ())
    }

    pub fn remove<Q>(&mut self, k: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.map.remove_exists(k)
    }

    pub fn contains<Q>(&self, k: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.map.contains_key(k)
    }

    pub fn first(&self) -> Option<&T> {
        self.map.front().map(|(k, _)| k)
    }

    pub fn last(&self) -> Option<&T> {
        self.map.back().map(|(k, _)| k)
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.map.pop_front().map(|(k, _)| k)
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.map.pop_back().map(|(k, _)| k)
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.map.keys().cloned().collect()
    }

    pub fn keys(&self) -> impl Iterator<Item = &T> {
        self.map.keys()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.map.keys()
    }
}

impl<T> Default for OrderedSet<T>
where
    T: Eq + Hash + Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Debug for OrderedSet<T>
where
    T: fmt::Debug + Eq + Hash + Clone,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.map.keys()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::OrderedSet;

    #[test]
    fn test_ordered_set_order_and_pop() {
        let mut s = OrderedSet::new();
        assert!(s.insert("a"));
        assert!(s.insert("b"));
        assert!(!s.insert("a"));
        assert_eq!(s.first(), Some(&"a"));
        assert_eq!(s.last(), Some(&"b"));
        assert_eq!(s.to_vec(), vec!["a", "b"]);
        assert_eq!(s.pop_front(), Some("a"));
        assert_eq!(s.pop_back(), Some("b"));
        assert!(s.is_empty());
    }
}
