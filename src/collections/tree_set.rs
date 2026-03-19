use std::collections::BTreeSet;

/// Ordered set backed by a tree structure.
pub struct TreeSet<T>
where
    T: Ord,
{
    data: BTreeSet<T>,
}

impl<T> TreeSet<T>
where
    T: Ord,
{
    pub fn new() -> Self {
        Self {
            data: BTreeSet::new(),
        }
    }

    pub fn add(&mut self, e: T) {
        self.data.insert(e);
    }

    pub fn delete(&mut self, e: &T) -> bool {
        self.data.remove(e)
    }

    pub fn contains(&self, e: &T) -> bool {
        self.data.contains(e)
    }

    pub fn add_if_absent(&mut self, e: T) -> bool {
        self.data.insert(e)
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn iterate(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<T> Default for TreeSet<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> TreeSet<T>
where
    T: Ord + Clone,
{
    pub fn add_all<I>(&mut self, e: I)
    where
        I: IntoIterator<Item = T>,
    {
        for v in e {
            self.add(v);
        }
    }

    pub fn delete_all<'a, I>(&mut self, e: I)
    where
        I: IntoIterator<Item = &'a T>,
        T: 'a,
    {
        for v in e {
            self.delete(v);
        }
    }

    pub fn mutual_exclude(&self, another: &TreeSet<T>) -> bool {
        self.data.is_disjoint(&another.data)
    }

    pub fn intersect(&self, another: &TreeSet<T>) -> TreeSet<T> {
        TreeSet {
            data: self.data.intersection(&another.data).cloned().collect(),
        }
    }

    pub fn union(&self, another: &TreeSet<T>) -> TreeSet<T> {
        TreeSet {
            data: self.data.union(&another.data).cloned().collect(),
        }
    }

    pub fn union_inplace(&mut self, another: &TreeSet<T>) {
        for val in &another.data {
            self.data.insert(val.clone());
        }
    }

    pub fn is_super_set(&self, another: &TreeSet<T>) -> bool {
        self.data.is_superset(&another.data)
    }

    pub fn is_sub_set(&self, another: &TreeSet<T>) -> bool {
        self.data.is_subset(&another.data)
    }

    pub fn shallow_copy(&self) -> TreeSet<T> {
        TreeSet {
            data: self.data.clone(),
        }
    }

    pub fn subtract(&mut self, another: &TreeSet<T>) {
        for val in &another.data {
            self.data.remove(val);
        }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }

    pub fn equals(&self, another: &TreeSet<T>) -> bool {
        self.data == another.data
    }
}

#[cfg(test)]
mod tests {
    use super::TreeSet;

    #[test]
    fn test_basic_set_ops() {
        let mut s1 = TreeSet::new();
        s1.add_all([1, 2, 3]);
        let mut s2 = TreeSet::new();
        s2.add_all([3, 4]);

        assert!(s1.contains(&1));
        assert_eq!(s1.size(), 3);

        let i = s1.intersect(&s2).to_vec();
        assert_eq!(i, vec![3]);

        let u = s1.union(&s2).to_vec();
        assert_eq!(u, vec![1, 2, 3, 4]);

        s1.subtract(&s2);
        assert_eq!(s1.to_vec(), vec![1, 2]);
        assert!(s1.mutual_exclude(&s2));
    }
}
