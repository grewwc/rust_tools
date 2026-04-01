use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Mutex;

struct Skipnode<K, V> {
    k: std::mem::MaybeUninit<K>,
    v: std::mem::MaybeUninit<V>,
    forward: Vec<*mut Skipnode<K, V>>,
}

impl<K, V> Skipnode<K, V> {
    fn new(k: K, v: V, max_height: usize) -> *mut Self {
        let ret = Box::new(Skipnode {
            k: std::mem::MaybeUninit::new(k),
            v: std::mem::MaybeUninit::new(v),
            forward: vec![ptr::null_mut(); max_height],
        });
        Box::into_raw(ret) as *mut Self
    }
}

pub struct SkipMap<K, V> {
    head: Skipnode<K, V>,
    max_height: usize,
    len: usize,
    cmp: fn(&K, &K) -> i32,

    rng: Mutex<StdRng>,
}

// 安全地实现 Send + Sync
// SkipMap 内部使用裸指针，但如果 K 和 V 是 Send + Sync，且所有操作通过 &mut self 进行，
// 那么可以安全地实现 Send + Sync
unsafe impl<K: Send, V: Send> Send for SkipMap<K, V> {}
unsafe impl<K: Send + Sync, V: Send + Sync> Sync for SkipMap<K, V> {}

impl<K, V> SkipMap<K, V> {
    unsafe fn free_chain(mut curr: *mut Skipnode<K, V>) {
        while !curr.is_null() {
            let next = unsafe { (&*curr).forward[0] };
            unsafe {
                drop(Box::from_raw(curr));
            }
            curr = next;
        }
    }

    pub fn new(max_height: usize, cmp: fn(&K, &K) -> i32) -> Box<Self> {
        // 使用随机种子创建 StdRng
        let seed: [u8; 32] = rand::random();
        let ret = Box::new(SkipMap {
            head: Skipnode {
                k: std::mem::MaybeUninit::uninit(),
                v: std::mem::MaybeUninit::uninit(),
                forward: vec![ptr::null_mut(); max_height],
            },
            max_height,
            len: 0,
            cmp,
            rng: Mutex::new(StdRng::from_seed(seed)),
        });
        ret
    }

    pub fn insert(&mut self, k: K, v: V) {
        let level = self.level().min(self.max_height - 1);
        unsafe {
            let (updates, found) = self.find(&k, level);
            let found = found as *mut Skipnode<K, V>;
            if !found.is_null() {
                (&mut *found).v.write(v);
                return;
            }
            let new_node = Skipnode::new(k, v, self.max_height);

            for i in (0..=level).rev() {
                let prev = *updates.get_unchecked(i) as *mut Skipnode<K, V>;
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

    pub fn get_ref(&self, k: &K) -> Option<&V> {
        let (_, found) = self.find(k, self.max_height - 1);
        if found == ptr::null_mut() {
            return None;
        }
        Some(unsafe { (&*found).v.assume_init_ref() })
    }

    pub fn get(&self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        self.get_ref(k).cloned()
    }

    pub fn contains_key(&self, k: &K) -> bool {
        self.get_ref(k).is_some()
    }

    pub fn contains(&self, k: &K) -> bool {
        self.contains_key(k)
    }

    pub fn range(&self, r: std::ops::Range<K>) -> Vec<(&K, &V)> {
        let start = r.start;
        let end = r.end;
        let mut prev = &self.head as *const Skipnode<K, V>;
        let mut ret = vec![];
        let f = self.cmp;
        unsafe {
            for i in (0..self.max_height).rev() {
                let mut curr = *(&*prev).forward.get_unchecked(i);
                while !curr.is_null() && f((*curr).k.assume_init_ref(), &start) < 0 {
                    prev = curr;
                    curr = (&*curr).forward[i];
                }
            }
            let mut curr = *(&*prev).forward.get_unchecked(0);
            while !curr.is_null() {
                if f((*curr).k.assume_init_ref(), &end) >= 0 {
                    break;
                }
                let k = (*curr).k.assume_init_ref();
                let v = (*curr).v.assume_init_ref();
                ret.push((k, v));
                curr = (&*curr).forward[0];
            }
        }
        ret
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
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
        if found.is_null() {
            return false;
        }
        unsafe {
            for i in 0..self.max_height {
                let prev = *updates.get_unchecked(i) as *mut Skipnode<K, V>;
                let next = *(&*prev).forward.get_unchecked(i);
                if next.is_null() || found != next {
                    continue;
                }
                (&mut *prev).forward[i] = *(&*next).forward.get_unchecked(i);
            }
            drop(Box::from_raw(found as *mut Skipnode<K, V>));
        }
        self.len -= 1;
        true
    }
}

impl<K, V> Drop for SkipMap<K, V> {
    fn drop(&mut self) {
        let first = self
            .head
            .forward
            .first()
            .copied()
            .unwrap_or(ptr::null_mut());
        unsafe {
            SkipMap::free_chain(first);
        }
    }
}

pub struct SkipListIter<'a, K, V> {
    curr: *mut Skipnode<K, V>,
    _marker: PhantomData<&'a SkipMap<K, V>>,
}

impl<'a, K, V> IntoIterator for &'a SkipMap<K, V> {
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

impl<'a, K, V> Iterator for SkipListIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr.is_null() {
            return None;
        }
        unsafe {
            let node = &*self.curr;
            self.curr = *node.forward.get_unchecked(0);
            Some((node.k.assume_init_ref(), node.v.assume_init_ref()))
        }
    }
}

impl<K, V> SkipMap<K, V> {
    fn find(&self, k: &K, level: usize) -> (Vec<*const Skipnode<K, V>>, *const Skipnode<K, V>) {
        let f = self.cmp;
        unsafe {
            let mut updates = vec![ptr::null(); level + 1];
            let mut prev = &self.head as *const Skipnode<K, V>;
            let mut found = ptr::null_mut();
            for i in (0..=level).rev() {
                let mut curr = *(&*prev).forward.get_unchecked(i);
                while curr != ptr::null_mut() && f((&*curr).k.assume_init_ref(), &k) < 0 {
                    prev = curr;
                    curr = *((&*curr).forward).get_unchecked(i);
                }
                if curr != ptr::null_mut() && f((*curr).k.assume_init_ref(), &k) == 0 {
                    found = curr;
                }
                updates[i] = prev;
            }
            (updates, found)
        }
    }

    // private functions
    fn level(&self) -> usize {
        const PROB: f32 = 0.5;
        let mut ret = 0_usize;
        let max = self.max_height.saturating_sub(1);
        while ret < max
            && self
                .rng
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .random::<f32>()
                < PROB
        {
            ret += 1;
        }
        ret
    }
}

pub struct SkipSet<T>
where
    T: Ord,
{
    inner: Box<SkipMap<T, ()>>,
}

// SkipSet 的 Send + Sync 实现
unsafe impl<T: Ord + Send> Send for SkipSet<T> {}
unsafe impl<T: Ord + Send + Sync> Sync for SkipSet<T> {}

impl<T> SkipSet<T>
where
    T: Ord,
{
    pub fn new(max_height: usize) -> Self {
        Self {
            inner: SkipMap::new(max_height, Self::cmp),
        }
    }

    fn cmp(a: &T, b: &T) -> i32 {
        a.cmp(b) as i32
    }

    pub fn insert(&mut self, value: T) -> bool {
        if self.contains(&value) {
            return false;
        }
        self.inner.insert(value, ());
        true
    }

    pub fn remove(&mut self, value: &T) -> bool {
        self.inner.remove(value)
    }

    pub fn range(&self, r: std::ops::Range<T>) -> Vec<&T> {
        self.inner.range(r).into_iter().map(|(k, _)| k).collect()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn contains(&self, value: &T) -> bool {
        self.inner.contains_key(value)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (&*self.inner).into_iter().map(|(k, _)| k)
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_list(max_height: usize) -> Box<SkipMap<i32, i32>> {
        SkipMap::new(max_height, |a: &i32, b: &i32| a.cmp(b) as i32)
    }

    #[test]
    fn insert_and_get_single() {
        let mut list = new_list(4);
        list.insert(1, 100);
        assert_eq!(list.get(&1), Some(100));
    }

    #[test]
    fn insert_and_get_multiple() {
        let mut list = new_list(4);
        list.insert(1, 100);
        list.insert(2, 200);
        list.insert(3, 300);
        assert_eq!(list.get(&2), Some(200));
        assert_eq!(list.get(&1), Some(100));
        assert_eq!(list.get(&3), Some(300));
        assert_eq!(list.get(&4), None);
    }

    #[test]
    fn update_existing_key() {
        let mut list = new_list(4);
        list.insert(1, 100);
        list.insert(1, 999);
        assert_eq!(list.get(&1), Some(999));
    }

    #[test]
    fn remove_existing_key() {
        let mut list = new_list(4);
        list.insert(1, 100);
        list.insert(2, 200);
        assert!(list.remove(&1));
        assert_eq!(list.get(&1), None);
        assert_eq!(list.get(&2), Some(200));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn remove_nonexistent_key() {
        let mut list = new_list(4);
        list.insert(1, 100);
        assert!(!list.remove(&999));
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn clear_all() {
        let mut list = new_list(4);
        list.insert(1, 100);
        list.insert(2, 200);
        list.insert(3, 300);
        assert_eq!(list.len(), 3);
        list.clear();
        assert_eq!(list.len(), 0);
        assert_eq!(list.get(&1), None);
    }

    #[test]
    fn range_query() {
        let mut list = new_list(4);
        for i in 1..=10 {
            list.insert(i, i * 100);
        }
        let range = list.range(3..7);
        assert_eq!(range.len(), 4);
        assert_eq!(range[0], (&3, &300));
        assert_eq!(range[3], (&6, &600));
    }

    #[test]
    fn iteration() {
        let mut list = new_list(4);
        list.insert(3, 300);
        list.insert(1, 100);
        list.insert(2, 200);
        let items: Vec<_> = (&list).into_iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (&1, &100));
        assert_eq!(items[1], (&2, &200));
        assert_eq!(items[2], (&3, &300));
    }

    #[test]
    fn skipset_basic() {
        let mut set = SkipSet::new(4);
        assert!(set.insert(1));
        assert!(set.insert(2));
        assert!(!set.insert(1)); // duplicate
        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(!set.contains(&3));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn skipset_remove() {
        let mut set = SkipSet::new(4);
        set.insert(1);
        set.insert(2);
        set.insert(3);
        assert!(set.remove(&2));
        assert!(!set.remove(&2)); // already removed
        assert!(!set.contains(&2));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn skipset_iteration() {
        let mut set = SkipSet::new(4);
        set.insert(3);
        set.insert(1);
        set.insert(2);
        let items: Vec<_> = set.iter().collect();
        assert_eq!(items, vec![&1, &2, &3]);
    }

    #[test]
    fn skipset_range() {
        let mut set = SkipSet::new(4);
        for i in 1..=10 {
            set.insert(i);
        }
        let range: Vec<_> = set.range(3..7);
        assert_eq!(range, vec![&3, &4, &5, &6]);
    }

    #[test]
    fn send_sync_compile_check() {
        // 这个测试确保 SkipMap 和 SkipSet 实现 Send + Sync
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<SkipMap<i32, i32>>();
        assert_sync::<SkipMap<i32, i32>>();
        assert_send::<SkipSet<i32>>();
        assert_sync::<SkipSet<i32>>();
    }
}
