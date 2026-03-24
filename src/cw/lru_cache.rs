use rustc_hash::FxHashMap;
use std::{
    cell::RefCell,
    fmt::Display,
    hash::Hash,
    rc::{Rc, Weak},
    time::{Duration, Instant},
};

struct Node<K, V>
where
    K: Hash + Eq + Default,
    V: Default + Display,
{
    key: K,
    val: V,
    expires_at: Option<Instant>,
    prev: Option<Weak<RefCell<Node<K, V>>>>,
    next: Option<Rc<RefCell<Node<K, V>>>>,
}

impl<K, V> Node<K, V>
where
    K: Eq + Default + Hash,
    V: Default + Display,
{
    fn new(k: K, v: V, expires_at: Option<Instant>) -> Self {
        Self {
            key: k,
            val: v,
            expires_at,
            prev: None,
            next: None,
        }
    }
}

pub struct LruCache<K, V>
where
    K: Eq + Hash + Default + Clone,
    V: Default + Display,
{
    map: FxHashMap<K, Rc<RefCell<Node<K, V>>>>,
    head: Rc<RefCell<Node<K, V>>>,
    tail: Rc<RefCell<Node<K, V>>>,
    len: usize,
    cap: usize,
    ttl_ms: i64,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Default + Clone,
    V: Display + Default,
{
    pub fn new(cap: usize) -> Self {
        let dummy_head = Rc::new(RefCell::new(Node::new(K::default(), V::default(), None)));
        let dummy_tail = Rc::new(RefCell::new(Node::new(K::default(), V::default(), None)));
        dummy_head.borrow_mut().next = Some(dummy_tail.clone());
        dummy_tail.borrow_mut().prev = Some(Rc::downgrade(&dummy_head));
        Self {
            map: FxHashMap::default(),
            head: dummy_head,
            tail: dummy_tail,
            len: 0usize,
            cap,
            ttl_ms: -1,
        }
    }

    pub fn with_ttl(cap: usize, ttl_ms: i64) -> Self {
        let mut c = Self::new(cap);
        c.ttl_ms = ttl_ms;
        c
    }

    pub fn set_ttl_ms(&mut self, ttl_ms: i64) {
        self.ttl_ms = ttl_ms;
        if self.ttl_ms >= 0 {
            self.purge_expired(Instant::now());
        }
    }

    pub fn ttl_ms(&self) -> i64 {
        self.ttl_ms
    }

    pub fn put(&mut self, k: K, v: V) {
        let now = Instant::now();
        if self.ttl_ms >= 0 {
            self.purge_expired(now);
        }

        if let Some(node) = self.map.get(&k).cloned() {
            if self.is_expired_rc(&node, now) {
                self.remove_node(node);
            } else {
                {
                    let mut n = node.borrow_mut();
                    n.val = v;
                    n.expires_at = self.calc_expires_at(now);
                }
                self.move_node_to_front(node);
                return;
            }
        }

        let new_node = Rc::new(RefCell::new(Node::new(
            k.clone(),
            v,
            self.calc_expires_at(now),
        )));
        if self.len == self.cap
            && let Some(removed) = self.remove_tail()
        {
            self.map.remove(&removed.borrow().key);
            self.len -= 1;
        }
        self.add_to_front(new_node.clone());
        self.map.insert(k, new_node);
        self.len += 1;
    }

    pub fn get(&mut self, k: K) -> Option<&V> {
        let now = Instant::now();
        if let Some(node) = self.map.get(&k).cloned() {
            if self.is_expired_rc(&node, now) {
                self.remove_node(node);
                return None;
            }
            let x = node.as_ptr();
            self.move_node_to_front(node);
            return unsafe { Some(&(*x).val) };
        }
        None
    }

    pub fn get_ref(&mut self, k: &K) -> Option<&V> {
        let now = Instant::now();
        if let Some(node) = self.map.get(k).cloned() {
            if self.is_expired_rc(&node, now) {
                self.remove_node(node);
                return None;
            }
            let x = node.as_ptr();
            self.move_node_to_front(node);
            return unsafe { Some(&(*x).val) };
        }
        None
    }

    pub fn len(&self) -> usize {
        if self.ttl_ms < 0 {
            return self.len;
        }
        let now = Instant::now();
        let mut count = 0usize;
        let mut curr = self.head.borrow().next.clone();
        while let Some(node) = curr {
            if Rc::ptr_eq(&node, &self.tail) {
                break;
            }
            let (expires_at, next) = {
                let n = node.borrow();
                (n.expires_at, n.next.clone())
            };
            if !Self::is_expired_at(expires_at, now) {
                count = count.saturating_add(1);
            }
            curr = next;
        }
        count
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn contains_key(&self, k: &K) -> bool {
        let Some(node) = self.map.get(k) else {
            return false;
        };
        if self.ttl_ms < 0 {
            return true;
        }
        !Self::is_expired_at(node.borrow().expires_at, Instant::now())
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.len = 0;
        self.head.borrow_mut().next = Some(self.tail.clone());
        self.tail.borrow_mut().prev = Some(Rc::downgrade(&self.head));
    }

    fn calc_expires_at(&self, now: Instant) -> Option<Instant> {
        if self.ttl_ms < 0 {
            return None;
        }
        now.checked_add(Duration::from_millis(self.ttl_ms as u64))
    }

    fn is_expired_rc(&self, node: &Rc<RefCell<Node<K, V>>>, now: Instant) -> bool {
        if self.ttl_ms < 0 {
            return false;
        }
        Self::is_expired_at(node.borrow().expires_at, now)
    }

    fn is_expired_at(expires_at: Option<Instant>, now: Instant) -> bool {
        matches!(expires_at, Some(t) if t <= now)
    }
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Default + Clone,
    V: Default + Display,
{
    fn move_node_to_front(&mut self, node: Rc<RefCell<Node<K, V>>>) {
        let curr_head = self.head.borrow_mut().next.take().unwrap();
        if Rc::ptr_eq(&curr_head, &node) {
            self.head.borrow_mut().next = Some(curr_head);
            return;
        }
        let next = node.borrow().next.clone().unwrap();
        let prev = node.borrow().prev.clone().unwrap().upgrade().unwrap();
        prev.borrow_mut().next = Some(next.clone());
        next.borrow_mut().prev = Some(Rc::downgrade(&prev));

        self.head.borrow_mut().next = Some(node.clone());
        node.borrow_mut().prev = Some(Rc::downgrade(&self.head));
        node.borrow_mut().next = Some(curr_head.clone());
        curr_head.borrow_mut().prev = Some(Rc::downgrade(&node));
    }

    fn add_to_front(&mut self, node: Rc<RefCell<Node<K, V>>>) {
        let prev_head = self.head.borrow_mut().next.take().unwrap();
        self.head.borrow_mut().next = Some(node.clone());
        node.borrow_mut().prev = Some(Rc::downgrade(&self.head));

        node.borrow_mut().next = Some(prev_head.clone());
        prev_head.borrow_mut().prev = Some(Rc::downgrade(&node));
        if self.len == 0 {
            self.tail.borrow_mut().prev = Some(Rc::downgrade(&node));
            node.borrow_mut().next = Some(self.tail.clone());
        }
    }

    fn remove_tail(&mut self) -> Option<Rc<RefCell<Node<K, V>>>> {
        if self.len == 0 {
            return None;
        }
        let prev = self
            .tail
            .borrow_mut()
            .prev
            .take()
            .unwrap()
            .upgrade()
            .unwrap();
        let prev_2 = prev.borrow_mut().prev.take().unwrap().upgrade().unwrap();
        prev_2.borrow_mut().next = Some(self.tail.clone());
        self.tail.borrow_mut().prev = Some(Rc::downgrade(&prev_2));
        Some(prev)
    }

    fn detach_node(&mut self, node: &Rc<RefCell<Node<K, V>>>) {
        let (prev, next) = {
            let n = node.borrow();
            (n.prev.as_ref().and_then(|w| w.upgrade()), n.next.clone())
        };
        let Some(prev) = prev else {
            return;
        };
        let Some(next) = next else {
            return;
        };
        prev.borrow_mut().next = Some(next.clone());
        next.borrow_mut().prev = Some(Rc::downgrade(&prev));
        node.borrow_mut().prev = None;
        node.borrow_mut().next = None;
    }

    fn remove_node(&mut self, node: Rc<RefCell<Node<K, V>>>) {
        if self.len == 0 || Rc::ptr_eq(&node, &self.head) || Rc::ptr_eq(&node, &self.tail) {
            return;
        }
        let key = node.borrow().key.clone();
        self.detach_node(&node);
        self.map.remove(&key);
        self.len -= 1;
    }

    fn purge_expired(&mut self, now: Instant) {
        if self.ttl_ms < 0 || self.len == 0 {
            return;
        }
        let mut curr = self.tail.borrow().prev.as_ref().and_then(|w| w.upgrade());
        while let Some(node) = curr {
            if Rc::ptr_eq(&node, &self.head) {
                break;
            }
            let (prev, expired, key) = {
                let n = node.borrow();
                (
                    n.prev.as_ref().and_then(|w| w.upgrade()),
                    Self::is_expired_at(n.expires_at, now),
                    n.key.clone(),
                )
            };
            if expired {
                self.detach_node(&node);
                self.map.remove(&key);
                self.len -= 1;
            }
            curr = prev;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LruCache;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_lru_cache_basic_and_clear() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        assert!(c.is_empty());
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.len(), 2);
        assert!(c.contains_key(&1));
        assert_eq!(c.get(1), Some(&10));
        c.put(3, 30);
        assert!(!c.contains_key(&2));
        assert!(c.contains_key(&3));

        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.get_ref(&1), None);
    }

    #[test]
    fn test_lru_cache_ttl_default_no_expire() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 10);
        sleep(Duration::from_millis(30));
        assert_eq!(c.get(1), Some(&10));
    }

    #[test]
    fn test_lru_cache_ttl_expire() {
        let mut c: LruCache<i32, i32> = LruCache::with_ttl(2, 20);
        c.put(1, 10);
        assert!(c.contains_key(&1));
        sleep(Duration::from_millis(40));
        assert!(!c.contains_key(&1));
        assert_eq!(c.get_ref(&1), None);
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_lru_cache_get_twice_front_safe() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.get(2), Some(&20));
        assert_eq!(c.get(2), Some(&20));
    }
}
