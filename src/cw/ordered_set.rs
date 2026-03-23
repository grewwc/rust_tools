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

    pub fn clear(&mut self) {
        self.map.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn insert(&mut self, k: T) -> bool {
        self.map.insert(k, ())
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
