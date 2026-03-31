use rand::{RngExt, rngs::ThreadRng};
use std::marker::PhantomData;
use std::ptr;

struct Skipnode<K, V>
where
    K: Clone,
    V: Clone,
{
    k: K,
    v: V,
    forward: Vec<*mut Skipnode<K, V>>,
}

impl<K, V> Skipnode<K, V>
where
    K: Clone,
    V: Clone,
{
    fn new(k: K, v: V, max_height: usize) -> *mut Self {
        let ret = Box::new(Skipnode {
            k,
            v,
            forward: vec![ptr::null_mut(); max_height],
        });
        Box::into_raw(ret) as *mut Self
    }
}

pub struct Skiplist<K, V>
where
    K: Clone,
    V: Clone,
{
    head: Skipnode<K, V>,
    max_height: usize,
    len: usize,
    cmp: Box<dyn Fn(&K, &K) -> i32>,

    rng: ThreadRng,
}

impl<K, V> Skiplist<K, V>
where
    K: Clone,
    V: Clone,
{
    unsafe fn free_chain(mut curr: *mut Skipnode<K, V>) {
        while curr != ptr::null_mut() {
            let next = unsafe { (&*curr).forward[0] };
            unsafe {
                drop(Box::from_raw(curr));
            }
            curr = next;
        }
    }

    pub fn new(max_height: usize, cmp: impl Fn(&K, &K) -> i32 + 'static) -> Box<Self> {
        let ret = Box::new(Skiplist {
            head: Skipnode {
                k: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                v: unsafe { std::mem::MaybeUninit::zeroed().assume_init() },
                forward: vec![ptr::null_mut(); max_height],
            },
            max_height,
            len: 0,
            cmp: Box::new(cmp),
            rng: rand::rng(),
        });
        ret
    }

    pub fn insert(&mut self, k: K, v: V) {
        let level = self.level().min(self.max_height - 1);
        unsafe {
            let (updates, found) = self.find(&k.clone(), level);
            if found != ptr::null_mut() {
                (&mut *found).v = v;
                return;
            }
            let new_node = Skipnode::new(k, v, self.max_height);

            for i in (0..=level).rev() {
                let prev = updates[i];
                if prev == &mut self.head as *mut Skipnode<K, V> {
                    let tmp = self.head.forward[i];
                    self.head.forward[i] = new_node;
                    (&mut *new_node).forward[i] = tmp;
                } else {
                    let next = (&*prev).forward[i];
                    (&mut *prev).forward[i] = new_node;
                    (&mut *new_node).forward[i] = next;
                }
            }
        }
        self.len += 1;
    }

    pub fn get(&mut self, k: &K) -> Option<V> {
        let (_, found) = self.find(k, self.max_height - 1);
        if found == ptr::null_mut() {
            return None;
        }
        Some(unsafe { (*found).v.clone() })
    }

    pub fn contains(&mut self, k: &K) -> bool {
        self.get(k).is_some()
    }

    pub fn range(&mut self, r: std::ops::Range<K>) -> Vec<(&K, &V)> {
        let start = r.start;
        let end = r.end;
        let mut prev = &self.head as *const Skipnode<K, V>;
        let mut ret = vec![];
        let f = self.cmp.as_ref();
        unsafe {
            for i in (0..self.max_height).rev() {
                let mut curr = (&*prev).forward[i];
                while curr != ptr::null_mut() && f(&(*curr).k, &start) < 0 {
                    prev = curr;
                    curr = (&*curr).forward[i];
                }
            }
            let mut curr = (&*prev).forward[0];
            while curr != ptr::null_mut() {
                if f(&(*curr).k, &end) >= 0 {
                    break;
                }
                let k = &(*curr).k;
                let v = &(*curr).v;
                ret.push((k, v));
                curr = (&*curr).forward[0];
            }
        }
        ret
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn clear(&mut self) {
        let mut forward = std::mem::take(&mut self.head.forward);
        let first = forward.first().copied().unwrap_or(ptr::null_mut());
        unsafe {
            Self::free_chain(first);
        }
        forward.fill(ptr::null_mut());
        self.head.forward = forward;
        self.len = 0;
    }

    pub fn remove(&mut self, k: &K) -> bool {
        let (updates, found) = self.find(k, self.max_height - 1);
        if found == ptr::null_mut() {
            return false;
        }
        unsafe {
            for i in 0..self.max_height {
                let prev = *updates.get_unchecked(i);
                let next = (&*prev).forward[i];
                if next == ptr::null_mut() || found != next {
                    continue;
                }
                (&mut *prev).forward[i] = *(&*next).forward.get_unchecked(i);
            }
            drop(Box::from_raw(found));
        }
        self.len -= 1;
        true
    }
}

impl<K, V> Drop for Skiplist<K, V>
where
    K: Clone,
    V: Clone,
{
    fn drop(&mut self) {
        let first = self.head.forward.first().copied().unwrap_or(ptr::null_mut());
        unsafe {
            Skiplist::free_chain(first);
        }
    }
}

pub struct SkipListIter<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    curr: *mut Skipnode<K, V>,
    _marker: PhantomData<&'a Skiplist<K, V>>,
}

impl<'a, K, V> IntoIterator for &'a Skiplist<K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a V);

    type IntoIter = SkipListIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        let curr = unsafe { *self.head.forward.get_unchecked(0) };
        SkipListIter {
            curr,
            _marker: PhantomData,
        }
    }
}

impl<'a, K, V> Iterator for SkipListIter<'a, K, V>
where
    K: Clone,
    V: Clone,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr == ptr::null_mut() {
            return None;
        }
        unsafe {
            let node = &*self.curr;
            self.curr = *node.forward.get_unchecked(0);
            Some((&node.k, &node.v))
        }
    }
}

impl<K, V> Skiplist<K, V>
where
    K: Clone,
    V: Clone,
{
    fn find(&mut self, k: &K, level: usize) -> (Vec<*mut Skipnode<K, V>>, *mut Skipnode<K, V>) {
        let f: &dyn Fn(&K, &K) -> i32 = self.cmp.as_ref();
        unsafe {
            let mut updates = vec![ptr::null_mut(); level + 1];
            let mut prev = &mut self.head as *mut Skipnode<K, V>;
            let mut found = ptr::null_mut();
            for i in (0..=level).rev() {
                let mut curr = (&*prev).forward[i];
                while curr != ptr::null_mut() && f(&(&*curr).k, &k) < 0 {
                    prev = curr;
                    curr = *((&*curr).forward).get_unchecked(i);
                }
                if curr != ptr::null_mut() && f(&(*curr).k, &k) == 0 {
                    found = curr;
                }
                updates[i] = prev;
            }
            (updates, found)
        }
    }

    // private functions
    fn level(&mut self) -> usize {
        const PROB: f32 = 0.5;
        let mut ret = 0_usize;
        let max = self.max_height.saturating_sub(1);
        while ret < max && self.rng.random::<f32>() < PROB {
            ret += 1;
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmp_i32_desc(a: &i32, b: &i32) -> i32 {
        (*b).cmp(a) as i32
    }

    fn new_list(max_height: usize) -> Box<Skiplist<i32, i32>> {
        Skiplist::new(max_height, |a: &i32, b: &i32| a.cmp(b) as i32)
    }

    #[test]
    fn insert_and_get_single() {
        let mut sl = new_list(1);
        sl.insert(1, 10);
        assert_eq!(sl.get(&1), Some(10));
        assert_eq!(sl.get(&2), None);
    }

    #[test]
    fn insert_overwrite_updates_value() {
        let mut sl = new_list(1);
        sl.insert(1, 10);
        sl.insert(1, 20);
        assert_eq!(sl.get(&1), Some(20));
    }

    #[test]
    fn remove_existing_returns_true_and_deletes() {
        let mut sl = new_list(1);
        sl.insert(1, 10);
        sl.insert(2, 20);
        assert!(sl.remove(&1));
        assert_eq!(sl.get(&1), None);
        assert_eq!(sl.get(&2), Some(20));
    }

    #[test]
    fn remove_missing_returns_false() {
        let mut sl = new_list(1);
        sl.insert(1, 10);
        assert!(!sl.remove(&2));
        assert_eq!(sl.get(&1), Some(10));
    }

    #[test]
    fn contains_reflects_membership() {
        let mut sl = new_list(1);
        assert!(!sl.contains(&1));
        sl.insert(1, 10);
        assert!(sl.contains(&1));
        assert!(!sl.contains(&2));
    }

    #[test]
    fn iter_yields_sorted_by_comparator_ascending() {
        let mut sl = new_list(1);
        sl.insert(3, 30);
        sl.insert(1, 10);
        sl.insert(2, 20);
        let got = (&*sl)
            .into_iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        assert_eq!(got, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn iter_yields_sorted_by_comparator_descending() {
        let mut sl: Box<Skiplist<i32, i32>> = Skiplist::new(1, cmp_i32_desc);
        sl.insert(1, 10);
        sl.insert(3, 30);
        sl.insert(2, 20);
        let got = (&*sl).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(got, vec![3, 2, 1]);
    }

    #[test]
    fn range_full_when_start_is_min() {
        let mut sl = new_list(1);
        for i in 1..=5 {
            sl.insert(i, i * 10);
        }
        let got = sl
            .range(1..6)
            .into_iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        assert_eq!(got, vec![(1, 10), (2, 20), (3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn range_subset_middle_interval() {
        let mut sl = new_list(1);
        for i in 1..=6 {
            sl.insert(i, i * 10);
        }
        let got = sl
            .range(3..6)
            .into_iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        assert_eq!(got, vec![(3, 30), (4, 40), (5, 50)]);
    }

    #[test]
    fn range_empty_when_start_exceeds_all_keys() {
        let mut sl = new_list(1);
        for i in 1..=3 {
            sl.insert(i, i * 10);
        }
        let got = sl.range(4..10);
        assert!(got.is_empty());
    }

    #[test]
    fn clear_resets_len_and_membership_and_allows_reuse() {
        let mut sl = new_list(4);
        for i in 1..=20 {
            sl.insert(i, i * 10);
        }
        assert_eq!(sl.len(), 20);
        assert!(sl.contains(&7));
        sl.clear();
        assert_eq!(sl.len(), 0);
        assert!(!sl.contains(&7));
        assert_eq!(sl.get(&7), None);
        assert!((&*sl).into_iter().next().is_none());
        assert!(sl.range(1..100).is_empty());

        for i in 50..=60 {
            sl.insert(i, i * 3);
        }
        assert_eq!(sl.len(), 11);
        assert_eq!(sl.get(&55), Some(165));
        let keys = (&*sl).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(keys, (50..=60).collect::<Vec<_>>());
    }

    #[test]
    fn clear_is_idempotent_and_remove_after_clear_is_safe() {
        let mut sl = new_list(8);
        for i in 1..=10 {
            sl.insert(i, i);
        }
        sl.clear();
        sl.clear();
        assert_eq!(sl.len(), 0);
        assert!(!sl.remove(&1));
        assert!(!sl.remove(&42));
        assert!((&*sl).into_iter().next().is_none());
    }

    #[test]
    fn insert_overwrite_does_not_change_len() {
        let mut sl = new_list(6);
        sl.insert(7, 10);
        sl.insert(7, 11);
        sl.insert(7, 12);
        assert_eq!(sl.len(), 1);
        assert_eq!(sl.get(&7), Some(12));
        let items = (&*sl).into_iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>();
        assert_eq!(items, vec![(7, 12)]);
    }

    #[test]
    fn remove_min_max_and_middle_keeps_structure_consistent() {
        let mut sl = new_list(12);
        for i in 1..=50 {
            sl.insert(i, i * 2);
        }
        assert!(sl.remove(&1));
        assert!(sl.remove(&50));
        assert!(sl.remove(&25));
        assert_eq!(sl.len(), 47);
        assert_eq!(sl.get(&1), None);
        assert_eq!(sl.get(&50), None);
        assert_eq!(sl.get(&25), None);
        assert_eq!(sl.get(&24), Some(48));
        assert_eq!(sl.get(&26), Some(52));

        let keys = (&*sl).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(keys.len(), 47);
        assert_eq!(keys.first().copied(), Some(2));
        assert_eq!(keys.last().copied(), Some(49));
        assert!(!keys.contains(&1));
        assert!(!keys.contains(&25));
        assert!(!keys.contains(&50));
    }

    #[test]
    fn range_inclusive_start_exclusive_end() {
        let mut sl = new_list(4);
        for i in 1..=10 {
            sl.insert(i, i);
        }
        let got = sl.range(3..7).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(got, vec![3, 4, 5, 6]);

        let got = sl.range(1..1).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert!(got.is_empty());

        let got = sl.range(10..11).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(got, vec![10]);
    }

    #[test]
    fn range_descending_comparator_uses_start_as_upper_and_end_as_lower() {
        let mut sl: Box<Skiplist<i32, i32>> = Skiplist::new(6, cmp_i32_desc);
        for i in 1..=10 {
            sl.insert(i, i * 10);
        }
        let got = sl.range(9..4).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(got, vec![9, 8, 7, 6, 5]);
    }

    #[test]
    fn interleaved_inserts_and_removes_match_expected_len_and_order() {
        let mut sl = new_list(10);
        for i in 1..=100 {
            sl.insert(i, i);
        }
        for i in (2..=100).step_by(2) {
            assert!(sl.remove(&i));
        }
        assert_eq!(sl.len(), 50);
        for i in (2..=100).step_by(2) {
            assert_eq!(sl.get(&i), None);
        }
        for i in (1..=99).step_by(2) {
            assert_eq!(sl.get(&i), Some(i));
        }

        for i in 101..=120 {
            sl.insert(i, i);
        }
        assert_eq!(sl.len(), 70);
        for i in (1..=99).step_by(2) {
            assert!(sl.remove(&i));
        }
        assert_eq!(sl.len(), 20);
        let keys = (&*sl).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert_eq!(keys, (101..=120).collect::<Vec<_>>());
    }

    #[test]
    fn iter_matches_sorted_unique_keys_after_many_updates() {
        let mut sl = new_list(16);
        for i in 1..=200 {
            sl.insert(i, i * 2);
        }
        for i in (1..=200).step_by(3) {
            sl.insert(i, i * 5);
        }
        for i in (1..=200).step_by(7) {
            assert!(sl.remove(&i));
        }

        let keys = (&*sl).into_iter().map(|(k, _)| *k).collect::<Vec<_>>();
        assert!(keys.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(keys.len(), sl.len());
        for &k in &keys {
            assert!(sl.contains(&k));
        }
        for i in (1..=200).step_by(7) {
            assert!(!sl.contains(&i));
        }
    }

    #[test]
    fn range_large_height_matches_filtered_iteration() {
        let mut sl = new_list(20);
        for i in 1..=300 {
            sl.insert(i, i * 10);
        }
        for i in (1..=300).step_by(11) {
            assert!(sl.remove(&i));
        }
        let got = sl
            .range(57..243)
            .into_iter()
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        let expected = (&*sl)
            .into_iter()
            .filter(|(k, _)| **k >= 57 && **k < 243)
            .map(|(k, v)| (*k, *v))
            .collect::<Vec<_>>();
        assert_eq!(got, expected);
    }
}
