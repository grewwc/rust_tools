use std::{borrow::Borrow, fmt, hash::Hash};

use crate::common::types::FastSet;

#[derive(Clone)]
pub struct OrderedSet<T> {
    order: Vec<T>,
    set: FastSet<T>,
}

impl<T> OrderedSet<T>
where
    T: Eq + Hash + Clone,
{
    pub fn new() -> Self {
        Self {
            order: Vec::new(),
            set: FastSet::default(),
        }
    }

    pub fn insert(&mut self, v: T) -> bool {
        if self.set.insert(v.clone()) {
            self.order.push(v);
            return true;
        }
        false
    }

    pub fn contains<Q>(&self, v: &Q) -> bool
    where
        T: Borrow<Q>,
        Q: Eq + Hash + ?Sized,
    {
        self.set.contains(v)
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.order.clone()
    }
}

impl OrderedSet<String> {
    pub fn insert_str(&mut self, v: &str) -> bool {
        self.insert(v.to_string())
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
    T: fmt::Debug + Eq + Hash,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_set().entries(self.set.iter()).finish()
    }
}
