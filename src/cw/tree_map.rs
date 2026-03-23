use std::collections::BTreeMap;
use std::ops::Bound::Included;

/// Ordered map backed by a tree structure.
pub struct TreeMap<K, V>
where
    K: Ord,
{
    data: BTreeMap<K, V>,
}

impl<K, V> TreeMap<K, V>
where
    K: Ord,
{
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.data.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.data.get_mut(key)
    }

    pub fn contains(&self, key: &K) -> bool {
        self.data.contains_key(key)
    }

    /// Returns true when key did not exist and insertion happened.
    pub fn put(&mut self, key: K, value: V) -> bool {
        self.data.insert(key, value).is_none()
    }

    /// Returns true when key did not exist and insertion happened.
    pub fn put_if_absent(&mut self, key: K, value: V) -> bool {
        match self.data.entry(key) {
            std::collections::btree_map::Entry::Vacant(v) => {
                v.insert(value);
                true
            }
            std::collections::btree_map::Entry::Occupied(_) => false,
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn delete(&mut self, key: &K) -> bool {
        self.data.remove(key).is_some()
    }

    pub fn delete_all<'a, I>(&mut self, keys: I)
    where
        I: IntoIterator<Item = &'a K>,
        K: 'a,
    {
        for key in keys {
            self.data.remove(key);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &K> {
        self.data.keys()
    }

    pub fn iter_entry(&self) -> impl Iterator<Item = (&K, &V)> {
        self.data.iter()
    }

    pub fn for_each_entry<F>(&self, mut f: F)
    where
        F: FnMut(&K, &V),
    {
        for (k, v) in &self.data {
            f(k, v);
        }
    }

    pub fn for_each<F>(&self, mut f: F)
    where
        F: FnMut(&K),
    {
        for k in self.data.keys() {
            f(k);
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn first_key(&self) -> Option<&K> {
        self.data.first_key_value().map(|(k, _)| k)
    }

    pub fn last_key(&self) -> Option<&K> {
        self.data.last_key_value().map(|(k, _)| k)
    }

    pub fn pop_first(&mut self) -> Option<(K, V)> {
        self.data.pop_first()
    }

    pub fn pop_last(&mut self) -> Option<(K, V)> {
        self.data.pop_last()
    }

    pub fn search_range<'a>(&'a self, lower: &'a K, upper: &'a K) -> Vec<&'a K> {
        if lower > upper {
            return Vec::new();
        }
        self.data
            .range((Included(lower), Included(upper)))
            .map(|(k, _)| k)
            .collect()
    }
}

impl<K, V> Default for TreeMap<K, V>
where
    K: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> TreeMap<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    pub fn get_or_default(&self, key: &K, default_val: V) -> V {
        self.data.get(key).cloned().unwrap_or(default_val)
    }

    pub fn keys(&self) -> Vec<K> {
        self.data.keys().cloned().collect()
    }

    pub fn values(&self) -> Vec<V> {
        self.data.values().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::TreeMap;

    #[test]
    fn test_put_get_delete() {
        let mut m = TreeMap::new();
        assert!(m.put("a", 1));
        assert!(!m.put("a", 2));
        assert_eq!(m.get(&"a"), Some(&2));
        assert!(m.contains(&"a"));
        assert!(!m.put_if_absent("a", 3));
        assert!(m.put_if_absent("b", 4));
        assert_eq!(m.size(), 2);

        assert!(m.delete(&"a"));
        assert!(!m.delete(&"a"));
        assert_eq!(m.get_or_default(&"missing", 9), 9);
    }

    #[test]
    fn test_first_last_pop() {
        let mut m = TreeMap::new();
        m.put(2, "b");
        m.put(1, "a");
        m.put(3, "c");
        assert_eq!(m.first_key(), Some(&1));
        assert_eq!(m.last_key(), Some(&3));
        assert_eq!(m.pop_first(), Some((1, "a")));
        assert_eq!(m.pop_last(), Some((3, "c")));
        assert_eq!(m.keys(), vec![2]);
    }

    #[test]
    fn test_search_range() {
        let mut m = TreeMap::new();
        m.put(1, "a");
        m.put(3, "b");
        m.put(5, "c");
        let ks = m.search_range(&2, &5);
        assert_eq!(ks, vec![&3, &5]);
    }
}
