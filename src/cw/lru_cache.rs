use rustc_hash::FxHashMap;
use std::{
    hash::Hash,
    sync::{Arc, Mutex, Weak},
    time::{Duration, Instant},
};

struct Node<K, V> {
    key: Option<Arc<K>>,
    val: Option<V>,
    expires_at: Option<Instant>,
    prev: Option<Weak<Mutex<Node<K, V>>>>,
    next: Option<Arc<Mutex<Node<K, V>>>>,
}

impl<K, V> Node<K, V>
where
    K: Eq + Hash,
{
    fn new(k: Arc<K>, v: V, expires_at: Option<Instant>) -> Self {
        Self {
            key: Some(k),
            val: Some(v),
            expires_at,
            prev: None,
            next: None,
        }
    }

    fn new_sentinel() -> Self {
        Self {
            key: None,
            val: None,
            expires_at: None,
            prev: None,
            next: None,
        }
    }
}

pub struct LruCache<K, V> {
    map: FxHashMap<Arc<K>, Arc<Mutex<Node<K, V>>>>,
    head: Arc<Mutex<Node<K, V>>>,
    tail: Arc<Mutex<Node<K, V>>>,
    len: usize,
    cap: usize,
    ttl_ms: i64,
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash,
{
    pub fn new(cap: usize) -> Self {
        let dummy_head = Arc::new(Mutex::new(Node::new_sentinel()));
        let dummy_tail = Arc::new(Mutex::new(Node::new_sentinel()));
        dummy_head.lock().unwrap().next = Some(dummy_tail.clone());
        dummy_tail.lock().unwrap().prev = Some(Arc::downgrade(&dummy_head));
        Self {
            map: FxHashMap::default(),
            head: dummy_head,
            tail: dummy_tail,
            len: 0,
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
            if self.is_expired(&node, now) {
                self.remove_node(node);
            } else {
                let expires_at = self.calc_expires_at(now);
                {
                    let mut n = node.lock().unwrap();
                    n.val = Some(v);
                    n.expires_at = expires_at;
                }
                self.move_node_to_front(node);
                return;
            }
        }

        let key = Arc::new(k);
        let new_node = Arc::new(Mutex::new(Node::new(key.clone(), v, self.calc_expires_at(now))));
        if self.len == self.cap {
            if let Some(removed) = self.remove_tail_node() {
                if let Some(rkey) = removed.lock().unwrap().key.clone() {
                    self.map.remove(&rkey);
                }
                self.len -= 1;
            }
        }
        self.add_to_front(new_node.clone());
        self.map.insert(key, new_node);
        self.len += 1;
    }
    pub fn get(&mut self, k: K) -> Option<V>
    where
        V: Clone,
    {
        let now = Instant::now();
        let key = Arc::new(k);
        let node = self.map.get(&key)?.clone();
        if self.is_expired(&node, now) {
            self.remove_node(node);
            return None;
        }
        let val = node.lock().unwrap().val.clone();
        self.move_node_to_front(node);
        val
    }

    pub fn get_ref(&mut self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        let now = Instant::now();
        let node = self.map.get(k)?.clone();
        if self.is_expired(&node, now) {
            self.remove_node(node);
            return None;
        }
        let val = node.lock().unwrap().val.clone();
        self.move_node_to_front(node);
        val
    }

    pub fn len(&self) -> usize {
        if self.ttl_ms < 0 {
            return self.len;
        }
        let now = Instant::now();
        let mut count = 0;
        let mut curr = self.head.lock().unwrap().next.clone();
        while let Some(node) = curr {
            if Arc::ptr_eq(&node, &self.tail) {
                break;
            }
            let (expires_at, next) = {
                let n = node.lock().unwrap();
                (n.expires_at, n.next.clone())
            };
            if !Self::is_expired_at(expires_at, now) {
                count += 1;
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
        !Self::is_expired_at(node.lock().unwrap().expires_at, Instant::now())
    }

    pub fn clear(&mut self) {
        self.map.clear();
        self.len = 0;
        self.head.lock().unwrap().next = Some(self.tail.clone());
        self.tail.lock().unwrap().prev = Some(Arc::downgrade(&self.head));
    }

    fn calc_expires_at(&self, now: Instant) -> Option<Instant> {
        if self.ttl_ms < 0 {
            return None;
        }
        now.checked_add(Duration::from_millis(self.ttl_ms as u64))
    }

    fn is_expired(&self, node: &Arc<Mutex<Node<K, V>>>, now: Instant) -> bool {
        if self.ttl_ms < 0 {
            return false;
        }
        Self::is_expired_at(node.lock().unwrap().expires_at, now)
    }

    fn is_expired_at(expires_at: Option<Instant>, now: Instant) -> bool {
        matches!(expires_at, Some(t) if t <= now)
    }

    fn move_node_to_front(&mut self, node: Arc<Mutex<Node<K, V>>>) {
        let curr_head = self.head.lock().unwrap().next.take().unwrap();
        if Arc::ptr_eq(&curr_head, &node) {
            self.head.lock().unwrap().next = Some(curr_head);
            return;
        }
        let (next, prev) = {
            let n = node.lock().unwrap();
            (n.next.clone(), n.prev.clone())
        };
        let next = next.unwrap();
        let prev = prev.unwrap().upgrade().unwrap();
        prev.lock().unwrap().next = Some(next.clone());
        next.lock().unwrap().prev = Some(Arc::downgrade(&prev));

        self.head.lock().unwrap().next = Some(node.clone());
        node.lock().unwrap().prev = Some(Arc::downgrade(&self.head));
        node.lock().unwrap().next = Some(curr_head.clone());
        curr_head.lock().unwrap().prev = Some(Arc::downgrade(&node));
    }

    fn add_to_front(&mut self, node: Arc<Mutex<Node<K, V>>>) {
        let prev_head = self.head.lock().unwrap().next.take().unwrap();
        self.head.lock().unwrap().next = Some(node.clone());
        node.lock().unwrap().prev = Some(Arc::downgrade(&self.head));

        node.lock().unwrap().next = Some(prev_head.clone());
        prev_head.lock().unwrap().prev = Some(Arc::downgrade(&node));
        if self.len == 0 {
            self.tail.lock().unwrap().prev = Some(Arc::downgrade(&node));
            node.lock().unwrap().next = Some(self.tail.clone());
        }
    }

    fn remove_tail_node(&mut self) -> Option<Arc<Mutex<Node<K, V>>>> {
        if self.len == 0 {
            return None;
        }
        let prev = self
            .tail
            .lock()
            .unwrap()
            .prev
            .take()
            .unwrap()
            .upgrade()
            .unwrap();
        let prev_2 = prev.lock().unwrap().prev.take().unwrap().upgrade().unwrap();
        prev_2.lock().unwrap().next = Some(self.tail.clone());
        self.tail.lock().unwrap().prev = Some(Arc::downgrade(&prev_2));
        Some(prev)
    }

    fn detach_node(&mut self, node: &Arc<Mutex<Node<K, V>>>) {
        let (prev, next) = {
            let n = node.lock().unwrap();
            (n.prev.as_ref().and_then(|w| w.upgrade()), n.next.clone())
        };
        let Some(prev) = prev else {
            return;
        };
        let Some(next) = next else {
            return;
        };
        prev.lock().unwrap().next = Some(next.clone());
        next.lock().unwrap().prev = Some(Arc::downgrade(&prev));
        node.lock().unwrap().prev = None;
        node.lock().unwrap().next = None;
    }

    fn remove_node(&mut self, node: Arc<Mutex<Node<K, V>>>) {
        if self.len == 0
            || Arc::ptr_eq(&node, &self.head)
            || Arc::ptr_eq(&node, &self.tail)
        {
            return;
        }
        let key = node.lock().unwrap().key.clone();
        self.detach_node(&node);
        if let Some(key) = key {
            self.map.remove(&key);
        }
        self.len -= 1;
    }

    fn purge_expired(&mut self, now: Instant) {
        if self.ttl_ms < 0 || self.len == 0 {
            return;
        }
        let mut curr = self
            .tail
            .lock()
            .unwrap()
            .prev
            .as_ref()
            .and_then(|w| w.upgrade());
        while let Some(node) = curr {
            if Arc::ptr_eq(&node, &self.head) {
                break;
            }
            let (prev, expired, key) = {
                let n = node.lock().unwrap();
                (
                    n.prev.as_ref().and_then(|w| w.upgrade()),
                    Self::is_expired_at(n.expires_at, now),
                    n.key.clone(),
                )
            };
            if expired {
                self.detach_node(&node);
                if let Some(key) = key {
                    self.map.remove(&key);
                }
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
        assert_eq!(c.get(1), Some(10));
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
        assert_eq!(c.get(1), Some(10));
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
        assert_eq!(c.get(2), Some(20));
        assert_eq!(c.get(2), Some(20));
    }

    #[test]
    fn test_lru_cache_without_display_bound() {
        #[derive(Hash, Eq, PartialEq)]
        struct Key(&'static str);

        // No Display derive needed
        let mut c: LruCache<Key, String> = LruCache::new(2);
        c.put(Key("a"), String::from("A"));
        c.put(Key("b"), String::from("B"));
        assert_eq!(c.get_ref(&Key("a")), Some(String::from("A")));

        c.put(Key("c"), String::from("C"));
        assert!(c.get_ref(&Key("b")).is_none());
        assert_eq!(c.get_ref(&Key("c")), Some(String::from("C")));
    }
}
