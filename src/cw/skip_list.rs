use serde::{Deserialize, Deserializer, Serialize, Serializer};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Mutex;

struct SkipNode<K, V> {
    k: std::mem::MaybeUninit<K>,
    v: std::mem::MaybeUninit<V>,
    forward: Vec<*mut SkipNode<K, V>>,
}

impl<K, V> SkipNode<K, V> {
    fn new(k: K, v: V, max_height: usize) -> *mut Self {
        let ret = Box::new(SkipNode {
            k: std::mem::MaybeUninit::new(k),
            v: std::mem::MaybeUninit::new(v),
            forward: vec![ptr::null_mut(); max_height],
        });
        Box::into_raw(ret) as *mut Self
    }
}

pub struct SkipMap<K, V> {
    head: SkipNode<K, V>,
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

// ============== Serde implementations ==============

impl<K, V> Serialize for SkipMap<K, V>
where
    K: Serialize,
    V: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        // 序列化为: (max_height, Vec<(K, V)>)
        let mut seq = serializer.serialize_seq(Some(self.len() + 1))?;
        // 第一个元素是 max_height
        seq.serialize_element(&self.max_height)?;
        // 后续元素是所有 key-value 对
        for (k, v) in self.iter() {
            seq.serialize_element(&(k, v))?;
        }
        seq.end()
    }
}

impl<'de, K, V> Deserialize<'de> for SkipMap<K, V>
where
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // 反序列化: (max_height, Vec<(K, V)>)
        let (max_height, items): (usize, Vec<(K, V)>) =
            Deserialize::deserialize(deserializer)?;
        // 使用默认比较器重建 SkipMap
        let mut map = *SkipMap::new(max_height, |a: &K, b: &K| a.cmp(b) as i32);
        for (k, v) in items {
            map.insert(k, v);
        }
        Ok(map)
    }
}

impl<K, V> SkipMap<K, V> {
    unsafe fn free_chain(mut curr: *mut SkipNode<K, V>) {
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
            head: SkipNode {
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
            let found = found as *mut SkipNode<K, V>;
            if !found.is_null() {
                (&mut *found).v.write(v);
                return;
            }
            let new_node = SkipNode::new(k, v, self.max_height);

            for i in (0..=level).rev() {
                let prev = *updates.get_unchecked(i) as *mut SkipNode<K, V>;
                if prev == &mut self.head as *mut SkipNode<K, V> {
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

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let (_, found) = self.find(k, self.max_height - 1);
        let found = found as *mut SkipNode<K, V>;
        if found == ptr::null_mut() {
            return None;
        }
        Some(unsafe { (&mut *found).v.assume_init_mut() })
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
        let mut prev = &self.head as *const SkipNode<K, V>;
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

    pub fn max_height(&self) -> usize {
        self.max_height
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
                let prev = *updates.get_unchecked(i) as *mut SkipNode<K, V>;
                let next = *(&*prev).forward.get_unchecked(i);
                if next.is_null() || found != next {
                    continue;
                }
                (&mut *prev).forward[i] = *(&*next).forward.get_unchecked(i);
            }
            drop(Box::from_raw(found as *mut SkipNode<K, V>));
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

impl<K, V> Clone for SkipMap<K, V>
where
    K: Clone + Ord,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut map = *SkipMap::new(self.max_height, self.cmp);
        for (k, v) in self.iter() {
            map.insert(k.clone(), v.clone());
        }
        map
    }
}

impl<K, V> FromIterator<(K, V)> for SkipMap<K, V>
where
    K: Ord,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        // 使用 max_height=16 作为默认值，通过解引用 Box 来创建
        let mut map = *SkipMap::new(16, |a: &K, b: &K| a.cmp(b) as i32);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}

pub struct IntoSkipMapIter<K, V> {
    curr_node: *mut SkipNode<K, V>,
}

impl<K, V> Drop for IntoSkipMapIter<K, V> {
    fn drop(&mut self) {
        unsafe {
            SkipMap::free_chain(self.curr_node);
        }
    }
}

impl<K, V> IntoIterator for Box<SkipMap<K, V>> {
    type Item = (K, V);
    type IntoIter = IntoSkipMapIter<K, V>;

    fn into_iter(mut self) -> Self::IntoIter {
        let first = self
            .head
            .forward
            .first()
            .copied()
            .unwrap_or(ptr::null_mut());
        self.head.forward.fill(ptr::null_mut());
        self.len = 0;
        IntoSkipMapIter { curr_node: first }
    }
}

impl<K, V> Iterator for IntoSkipMapIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr_node.is_null() {
            return None;
        }
        let temp = std::mem::replace(&mut self.curr_node, ptr::null_mut());
        unsafe {
            let boxed = Box::from_raw(temp);
            let next = (*boxed.forward.get_unchecked(0)) as *mut SkipNode<K, V>;
            self.curr_node = next;
            let k = ptr::read(&boxed.k).assume_init();
            let v = ptr::read(&boxed.v).assume_init();
            Some((k, v))
        }
    }
}

pub struct SkipListIter<'a, K, V> {
    curr: *mut SkipNode<K, V>,
    _marker: PhantomData<&'a SkipMap<K, V>>,
}

pub struct SkipListIterMut<'a, K, V> {
    curr: *mut SkipNode<K, V>,
    _marker: PhantomData<&'a mut SkipMap<K, V>>,
}

impl<'a, K, V> IntoIterator for &'a SkipMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = SkipListIter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

impl<'a, K, V> Iterator for SkipListIterMut<'a, K, V> {
    type Item = (&'a mut K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.curr.is_null() {
            return None;
        }
        unsafe {
            let node = &mut *self.curr;
            self.curr = *node.forward.get_unchecked(0);
            Some((node.k.assume_init_mut(), node.v.assume_init_mut()))
        }
    }
}

// FromIterator for Box<SkipMap<K, V>>
impl<K, V> FromIterator<(K, V)> for Box<SkipMap<K, V>>
where
    K: Ord,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = SkipMap::new(16, |a: &K, b: &K| a.cmp(b) as i32);
        for (k, v) in iter {
            map.insert(k, v);
        }
        map
    }
}
impl<K, V> SkipMap<K, V> {
    pub fn iter(&self) -> SkipListIter<'_, K, V> {
        let curr = unsafe { *self.head.forward.get_unchecked(0) };
        SkipListIter {
            curr,
            _marker: PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> SkipListIterMut<'_, K, V> {
        let curr = unsafe { *self.head.forward.get_unchecked(0) };
        SkipListIterMut {
            curr,
            _marker: PhantomData,
        }
    }

    fn find(&self, k: &K, level: usize) -> (Vec<*const SkipNode<K, V>>, *const SkipNode<K, V>) {
        let f = self.cmp;
        unsafe {
            let mut updates = vec![ptr::null(); level + 1];
            let mut prev = &self.head as *const SkipNode<K, V>;
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


// ============== SkipSet Serde implementations ==============

impl<T> Serialize for SkipSet<T>
where
    T: Serialize + Ord,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeSeq;
        // 序列化: (max_height, Vec<T>)
        let mut seq = serializer.serialize_seq(Some(self.len() + 1))?;
        seq.serialize_element(&self.inner.max_height())?;
        for v in self.iter() {
            seq.serialize_element(v)?;
        }
        seq.end()
    }
}

impl<'de, T> Deserialize<'de> for SkipSet<T>
where
    T: Deserialize<'de> + Ord,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (max_height, items): (usize, Vec<T>) =
            Deserialize::deserialize(deserializer)?;
        let mut set = SkipSet::new(max_height);
        for item in items {
            set.insert(item);
        }
        Ok(set)
    }
}

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

    /// Check if the set contains a value, accepting &str directly for String type.
    pub fn contains_str(&self, value: &str) -> bool
    where
        T: AsRef<str>,
    {
        for k in self.iter() {
            if k.as_ref() == value {
                return true;
            }
        }
        false
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn max_height(&self) -> usize {
        self.inner.max_height()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.inner.iter().map(|(k, _)| k)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.inner.iter_mut().map(|(k, _)| k)
    }

    /// Returns a reference to the first (smallest) element, or `None` if empty.
    pub fn first(&self) -> Option<&T> {
        self.inner.iter().next().map(|(k, _)| k)
    }

    /// Returns a reference to the last (largest) element, or `None` if empty.
    pub fn last(&self) -> Option<&T> {
        let mut iter = self.inner.iter();
        let mut last = iter.next()?;
        for item in iter {
            last = item;
        }
        Some(last.0)
    }

    /// Removes and returns the first (smallest) element, or `None` if empty.
    pub fn pop_first(&mut self) -> Option<T>
    where
        T: Clone,
    {
        let first = self.first()?.clone();
        if self.remove(&first) {
            Some(first)
        } else {
            None
        }
    }

    /// Removes and returns the last (largest) element, or `None` if empty.
    pub fn pop_last(&mut self) -> Option<T>
    where
        T: Clone,
    {
        let last = self.last()?.clone();
        if self.remove(&last) { Some(last) } else { None }
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }
}

impl<T> FromIterator<T> for Box<SkipSet<T>>
where
    T: Ord,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut set = SkipSet::new(16);
        for item in iter {
            set.insert(item);
        }
        Box::new(set)
    }
}

impl<T> Extend<T> for Box<SkipSet<T>>
where
    T: Ord,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for item in iter {
            self.insert(item);
        }
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
        let items: Vec<_> = list.iter().collect();
        assert_eq!(items.len(), 3);
        assert_eq!(items[0], (&1, &100));
        assert_eq!(items[1], (&2, &200));
        assert_eq!(items[2], (&3, &300));
    }

    #[test]
    fn into_iteration() {
        let mut list = new_list(4);
        list.insert(3, 300);
        list.insert(1, 100);
        list.insert(2, 200);
        let items: Vec<_> = list.into_iter().collect();
        assert_eq!(items, vec![(1, 100), (2, 200), (3, 300)]);
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

    #[test]
    fn skipmap_from_iter() {
        let data: Vec<(i32, &str)> = vec![(3, "c"), (1, "a"), (2, "b")];
        let map: Box<SkipMap<i32, &str>> = data.into_iter().collect();
        let items: Vec<_> = map.iter().collect();
        // SkipMap 按 key 排序
        assert_eq!(items, vec![(&1, &"a"), (&2, &"b"), (&3, &"c")]);
    }

    #[test]
    fn skipmap_clone() {
        let mut map = SkipMap::new(4, |a: &i32, b: &i32| a.cmp(b) as i32);
        map.insert(1, "one");
        map.insert(2, "two");
        map.insert(3, "three");
        
        let cloned = map.clone();
        assert_eq!(cloned.get_ref(&1), Some(&"one"));
        assert_eq!(cloned.get_ref(&2), Some(&"two"));
        assert_eq!(cloned.get_ref(&3), Some(&"three"));
        assert_eq!(cloned.len(), 3);
        
        // 原 map 不受影响
        map.insert(4, "four");
        assert_eq!(map.len(), 4);
        assert_eq!(cloned.len(), 3);
    }
}



