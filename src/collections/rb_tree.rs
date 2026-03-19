use std::collections::BTreeSet;
use std::ops::Bound::Included;

/// Ordered set-like tree with a red-black-tree style API.
///
/// Rust standard library uses a B-Tree internally for ordered sets.
/// This wrapper provides an API close to the original Go red-black tree usage.
pub struct RbTree<T>
where
    T: Ord,
{
    data: BTreeSet<T>,
}

impl<T> RbTree<T>
where
    T: Ord,
{
    pub fn new() -> Self {
        Self {
            data: BTreeSet::new(),
        }
    }

    pub fn contains(&self, val: &T) -> bool {
        self.data.contains(val)
    }

    pub fn search(&self, val: &T) -> Option<&T> {
        self.data.get(val)
    }

    pub fn insert(&mut self, val: T) -> bool {
        self.data.insert(val)
    }

    pub fn delete(&mut self, val: &T) -> bool {
        self.data.remove(val)
    }

    pub fn search_range<'a>(&'a self, lower: &'a T, upper: &'a T) -> Vec<&'a T> {
        if lower > upper {
            return Vec::new();
        }
        self.data
            .range((Included(lower), Included(upper)))
            .collect()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn min(&self) -> Option<&T> {
        self.data.first()
    }

    pub fn max(&self) -> Option<&T> {
        self.data.last()
    }
}

impl<T> Default for RbTree<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RbTree<T>
where
    T: Ord + Clone,
{
    /// Returns the stored value and whether insertion happened.
    pub fn search_or_insert(&mut self, val: T) -> (T, bool) {
        if let Some(existing) = self.data.get(&val) {
            return (existing.clone(), false);
        }
        self.data.insert(val.clone());
        (val, true)
    }
}

#[cfg(test)]
mod tests {
    use super::RbTree;

    #[test]
    fn test_insert_search_delete_range() {
        let mut tree = RbTree::new();
        assert!(tree.insert(5));
        assert!(tree.insert(1));
        assert!(tree.insert(10));
        assert!(!tree.insert(10));

        assert!(tree.contains(&5));
        assert_eq!(tree.search(&10), Some(&10));
        assert_eq!(tree.min(), Some(&1));
        assert_eq!(tree.max(), Some(&10));

        let range = tree.search_range(&2, &10);
        assert_eq!(range, vec![&5, &10]);

        assert!(tree.delete(&5));
        assert!(!tree.contains(&5));
        assert_eq!(tree.len(), 2);
    }

    #[test]
    fn test_search_or_insert() {
        let mut tree = RbTree::new();
        let (val, inserted) = tree.search_or_insert(3);
        assert_eq!(val, 3);
        assert!(inserted);

        let (val, inserted) = tree.search_or_insert(3);
        assert_eq!(val, 3);
        assert!(!inserted);
    }
}
