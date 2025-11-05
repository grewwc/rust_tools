use std::{collections::HashMap, hash::Hash};

pub struct UF<'a, T>
where
    T: Eq + Hash,
{
    id: HashMap<&'a T, &'a T>,
    size: HashMap<&'a T, usize>,

    n_groups: usize,
}

impl<'a, T> UF<'a, T>
where
    T: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            id: HashMap::new(),
            size: HashMap::new(),
            n_groups: 0,
        }
    }
}

impl<'a, T: Eq + Hash> UF<'a, T> {
    pub fn union(&mut self, v1: &'a T, v2: &'a T) -> bool {
        if !self.id.contains_key(v1) {
            self.id.insert(v1, v1);
            self.size.insert(v1, 1);
            self.n_groups += 1;
        }
        if !self.id.contains_key(v2) {
            self.id.insert(v2, v2);
            self.size.insert(v2, 1);
            self.n_groups += 1;
        }

        let r1 = self.find_root(v1);
        let r2 = self.find_root(v2);
        if r1 == r2 {
            return false;
        }
        let s1 = self.size.get(v1).unwrap();
        let s2 = self.size.get(v2).unwrap();
        if s1 < s2 {
            self.id.insert(r1, r2);
            self.size.insert(r2, s1 + s2);
        } else {
            self.id.insert(r2, r1);
            self.size.insert(r1, s1 + s2);
        }
        self.n_groups -= 1;
        true
    }

    pub fn find_root(&mut self, v: &'a T) -> &'a T {
        let mut root = v;
        loop {
            let parent = self.id.get(root).map(|x| *x).unwrap();
            if root == parent {
                break;
            }
            root = parent;
        }

        let mut id = v;
        while id != root {
            self.id.insert(id, root);
            self.size.insert(
                root,
                self.size.get(root).unwrap() + self.size.get(id).unwrap(),
            );
            id = self.id.get(id).map(|x| *x).unwrap();
        }

        root
    }

    pub fn is_connected(&mut self, v1: &'a T, v2: &'a T) -> bool {
        if !self.id.contains_key(v1) || !self.id.contains_key(v2) {
            return false;
        }
        self.find_root(v1) == self.find_root(v2)
    }

    pub fn n_groups(&self) -> usize {
        self.n_groups
    }
}

#[cfg(test)]
mod tests {
    use crate::collections::uf::UF;

    #[test]
    fn test_union() {
        let mut uf: UF<i32> = UF::new();
        let v1 = 1;
        let v2 = 2;
        let v3 = 3;
        assert!(uf.union(&v1, &v2));
        assert!(uf.is_connected(&v1, &v2));
        assert!(!uf.is_connected(&v1, &v3));
        assert!(!uf.union(&v1, &v2));
        assert!(uf.union(&v1, &v3));
        assert!(uf.is_connected(&v2, &v3));
        assert_eq!(1, uf.n_groups());
    }
}

