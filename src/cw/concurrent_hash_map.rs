use std::hash::{Hash, Hasher};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;

use rustc_hash::FxHasher;

use crate::commonw::types::FastMap;

pub struct ValueRef<'a, K, V> {
    _guard: RwLockReadGuard<'a, FastMap<K, V>>,
    value: NonNull<V>,
}

impl<K, V> Deref for ValueRef<'_, K, V> {
    type Target = V;

    fn deref(&self) -> &Self::Target {
        unsafe { self.value.as_ref() }
    }
}

pub struct ConcurrentHashMap<K, V>
where
    K: Eq + Hash,
{
    shards: Vec<RwLock<FastMap<K, V>>>,
    shard_mask: usize,
}

impl<K, V> ConcurrentHashMap<K, V>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self::with_shard_count(64)
    }

    pub fn with_shard_count(shard_count: usize) -> Self {
        let n = shard_count.max(1).next_power_of_two();
        let mut shards = Vec::with_capacity(n);
        for _ in 0..n {
            shards.push(RwLock::new(FastMap::default()));
        }
        Self {
            shards,
            shard_mask: n - 1,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let shard_count = 64usize;
        let mut map = Self::with_shard_count(shard_count);
        let per = (capacity / map.shards.len()).saturating_add(1);
        for shard in &mut map.shards {
            if let Ok(mut guard) = shard.write() {
                *guard = FastMap::with_capacity_and_hasher(per, Default::default());
            }
        }
        map
    }

    pub fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut guard) = shard.write() {
                guard.clear();
            }
        }
    }

    pub fn len(&self) -> usize {
        let mut sum = 0usize;
        for shard in &self.shards {
            if let Ok(guard) = shard.read() {
                sum = sum.saturating_add(guard.len());
            }
        }
        sum
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains_key(&self, key: &K) -> bool {
        let shard = self.shard_for_key(key);
        let guard = shard.read().unwrap();
        guard.contains_key(key)
    }

    pub fn get_cloned(&self, key: &K) -> Option<V>
    where
        V: Clone,
    {
        let shard = self.shard_for_key(key);
        let guard = shard.read().unwrap();
        guard.get(key).cloned()
    }

    pub fn get(&self, key: &K) -> Option<ValueRef<'_, K, V>> {
        let shard = self.shard_for_key(key);
        let guard = shard.read().unwrap();
        let value = match guard.get(key) {
            Some(v) => NonNull::from(v),
            None => return None,
        };
        Some(ValueRef {
            _guard: guard,
            value,
        })
    }

    pub fn get_ref(&self, key: &K) -> Option<ValueRef<'_, K, V>> {
        self.get(key)
    }

    pub fn get_with<R>(&self, key: &K, f: impl FnOnce(&V) -> R) -> Option<R> {
        let shard = self.shard_for_key(key);
        let guard = shard.read().unwrap();
        guard.get(key).map(f)
    }

    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let shard = self.shard_for_key(&key);
        let mut guard = shard.write().unwrap();
        guard.insert(key, value)
    }

    pub fn remove(&self, key: &K) -> Option<V> {
        let shard = self.shard_for_key(key);
        let mut guard = shard.write().unwrap();
        guard.remove(key)
    }

    pub fn put_if_absent(&self, key: K, value: V) -> Option<V>
    where
        V: Clone,
    {
        let shard = self.shard_for_key(&key);
        let mut guard = shard.write().unwrap();
        if let Some(existing) = guard.get(&key) {
            return Some(existing.clone());
        }
        guard.insert(key, value);
        None
    }

    pub fn get_or_insert_with(&self, key: K, f: impl FnOnce() -> V) -> V
    where
        V: Clone,
    {
        let shard = self.shard_for_key(&key);
        let mut guard = shard.write().unwrap();
        if let Some(existing) = guard.get(&key) {
            return existing.clone();
        }
        let v = f();
        guard.insert(key, v.clone());
        v
    }

    pub fn compute<R>(&self, key: K, f: impl FnOnce(Option<&V>) -> (Option<V>, R)) -> R {
        let shard = self.shard_for_key(&key);
        let mut guard = shard.write().unwrap();
        let current = guard.get(&key);
        let (next, out) = f(current);
        match next {
            Some(v) => {
                guard.insert(key, v);
            }
            None => {
                guard.remove(&key);
            }
        }
        out
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    fn shard_for_key(&self, key: &K) -> &RwLock<FastMap<K, V>> {
        let idx = self.shard_index(key);
        &self.shards[idx]
    }

    fn shard_index(&self, key: &K) -> usize {
        let mut h = FxHasher::default();
        key.hash(&mut h);
        (h.finish() as usize) & self.shard_mask
    }
}

impl<K, V> Default for ConcurrentHashMap<K, V>
where
    K: Eq + Hash,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;

    use super::ConcurrentHashMap;

    #[test]
    fn test_new_default_shard_count() {
        let m: ConcurrentHashMap<usize, usize> = ConcurrentHashMap::new();
        assert_eq!(m.shard_count(), 64);
    }

    #[test]
    fn test_with_shard_count_rounds_to_power_of_two() {
        let m: ConcurrentHashMap<usize, usize> = ConcurrentHashMap::with_shard_count(3);
        assert_eq!(m.shard_count(), 4);
    }

    #[test]
    fn test_with_shard_count_min_one() {
        let m: ConcurrentHashMap<usize, usize> = ConcurrentHashMap::with_shard_count(0);
        assert_eq!(m.shard_count(), 1);
    }

    #[test]
    fn test_shard_count_is_power_of_two() {
        let m: ConcurrentHashMap<usize, usize> = ConcurrentHashMap::with_shard_count(1000);
        let n = m.shard_count();
        assert_ne!(n, 0);
        assert_eq!(n & (n - 1), 0);
    }

    #[test]
    fn test_put_get_remove() {
        let m = ConcurrentHashMap::with_shard_count(8);
        assert!(m.is_empty());
        assert_eq!(m.insert("a", 1), None);
        assert_eq!(m.get_cloned(&"a"), Some(1));
        assert_eq!(m.insert("a", 2), Some(1));
        assert_eq!(m.get_cloned(&"a"), Some(2));
        assert_eq!(m.remove(&"a"), Some(2));
        assert_eq!(m.get_cloned(&"a"), None);
    }

    #[test]
    fn test_put_if_absent() {
        let m = ConcurrentHashMap::new();
        assert_eq!(m.put_if_absent("k", 1), None);
        assert_eq!(m.put_if_absent("k", 2), Some(1));
        assert_eq!(m.get_cloned(&"k"), Some(1));
    }

    #[test]
    fn test_contains_key() {
        let m = ConcurrentHashMap::new();
        assert!(!m.contains_key(&"k"));
        m.insert("k", 1);
        assert!(m.contains_key(&"k"));
    }

    #[test]
    fn test_get_with_maps_value() {
        let m = ConcurrentHashMap::new();
        m.insert("k", 7);
        let x = m.get_with(&"k", |v| v + 1);
        assert_eq!(x, Some(8));
    }

    #[test]
    fn test_get_with_missing() {
        let m: ConcurrentHashMap<&'static str, usize> = ConcurrentHashMap::new();
        let x = m.get_with(&"missing", |v| v + 1);
        assert_eq!(x, None);
    }

    #[test]
    fn test_get_by_ref_deref() {
        let m = ConcurrentHashMap::new();
        m.insert("k", vec![1, 2, 3]);
        let v = m.get(&"k").unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1);
    }

    #[test]
    fn test_get_by_ref_blocks_writer_until_drop() {
        let m = Arc::new(ConcurrentHashMap::with_shard_count(1));
        m.insert("k", 1usize);
        let guard = m.get(&"k").unwrap();

        let (started_tx, started_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let m2 = Arc::clone(&m);
        thread::spawn(move || {
            started_tx.send(()).ok();
            m2.remove(&"k");
            done_tx.send(()).ok();
        });

        started_rx.recv().unwrap();
        assert!(done_rx.try_recv().is_err());
        drop(guard);
        done_rx.recv().unwrap();
        assert_eq!(m.get_cloned(&"k"), None);
    }

    #[test]
    fn test_remove_missing() {
        let m: ConcurrentHashMap<&'static str, usize> = ConcurrentHashMap::new();
        assert_eq!(m.remove(&"k"), None);
    }

    #[test]
    fn test_clear() {
        let m = ConcurrentHashMap::new();
        for i in 0..1000 {
            m.insert(i, i + 1);
        }
        assert_eq!(m.len(), 1000);
        m.clear();
        assert_eq!(m.len(), 0);
        assert!(m.is_empty());
    }

    #[test]
    fn test_multiple_keys_len() {
        let m = ConcurrentHashMap::with_shard_count(16);
        for i in 0..100 {
            assert_eq!(m.insert(i, i), None);
        }
        assert_eq!(m.len(), 100);
        for i in 0..100 {
            assert_eq!(m.get_cloned(&i), Some(i));
        }
    }

    #[test]
    fn test_get_or_insert_with() {
        let m = ConcurrentHashMap::new();
        let v1 = m.get_or_insert_with("k", || 10);
        let v2 = m.get_or_insert_with("k", || 20);
        assert_eq!(v1, 10);
        assert_eq!(v2, 10);
        assert_eq!(m.get_cloned(&"k"), Some(10));
    }

    #[test]
    fn test_compute() {
        let m = ConcurrentHashMap::new();
        let out = m.compute("k", |cur| {
            let next = cur.copied().unwrap_or(0) + 1;
            (Some(next), next)
        });
        assert_eq!(out, 1);
        assert_eq!(m.get_cloned(&"k"), Some(1));

        let out = m.compute("k", |cur| {
            let next = cur.copied().unwrap_or(0) + 1;
            (Some(next), next)
        });
        assert_eq!(out, 2);
        assert_eq!(m.get_cloned(&"k"), Some(2));

        let removed = m.compute("k", |_| (None, true));
        assert!(removed);
        assert_eq!(m.get_cloned(&"k"), None);
    }

    #[test]
    fn test_compute_return_value() {
        let m = ConcurrentHashMap::new();
        let out = m.compute("k", |_| (Some(123), "ok"));
        assert_eq!(out, "ok");
        assert_eq!(m.get_cloned(&"k"), Some(123));
    }

    #[test]
    fn test_with_capacity_smoke() {
        let m = ConcurrentHashMap::with_capacity(10_000);
        for i in 0..5000 {
            m.insert(i, i + 1);
        }
        assert_eq!(m.len(), 5000);
    }

    #[test]
    fn test_concurrent_inserts() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let threads = 8usize;
        let per = 10_000usize;

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                let base = t * per;
                for i in 0..per {
                    let k = base + i;
                    m.insert(k, k + 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(m.len(), threads * per);
        assert_eq!(m.get_cloned(&0), Some(1));
        assert_eq!(m.get_cloned(&(threads * per - 1)), Some(threads * per));
    }

    #[test]
    fn test_concurrent_put_if_absent_single_winner() {
        let m = Arc::new(ConcurrentHashMap::<&'static str, usize>::with_shard_count(
            32,
        ));
        let threads = 32usize;
        let barrier = Arc::new(Barrier::new(threads));
        let results = Arc::new(Mutex::new(Vec::new()));

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            let results = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let r = m.put_if_absent("k", t);
                results.lock().unwrap().push(r);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        let final_v = m.get_cloned(&"k").unwrap();
        let results = results.lock().unwrap();
        let none_cnt = results.iter().filter(|x| x.is_none()).count();
        assert_eq!(none_cnt, 1);
        for r in results.iter().filter_map(|x| *x) {
            assert_eq!(r, final_v);
        }
    }

    #[test]
    fn test_concurrent_get_or_insert_with_only_calls_once() {
        let m = Arc::new(ConcurrentHashMap::<&'static str, usize>::with_shard_count(
            32,
        ));
        let threads = 64usize;
        let barrier = Arc::new(Barrier::new(threads));
        let calls = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            let calls = Arc::clone(&calls);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let v = m.get_or_insert_with("k", || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    42
                });
                assert_eq!(v, 42);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.get_cloned(&"k"), Some(42));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_concurrent_compute_increment_same_key() {
        let m = Arc::new(ConcurrentHashMap::<&'static str, usize>::with_shard_count(
            64,
        ));
        let threads = 32usize;
        let iters = 2000usize;
        let barrier = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for _ in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for _ in 0..iters {
                    m.compute("k", |cur| {
                        let next = cur.copied().unwrap_or(0) + 1;
                        (Some(next), ())
                    });
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.get_cloned(&"k"), Some(threads * iters));
    }

    #[test]
    fn test_concurrent_compute_disjoint_keys() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let threads = 16usize;
        let keys_per_thread = 200usize;
        let barrier = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let base = t * keys_per_thread;
                for i in 0..keys_per_thread {
                    let k = base + i;
                    m.compute(k, |cur| {
                        let next = cur.copied().unwrap_or(0) + 1;
                        (Some(next), ())
                    });
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.len(), threads * keys_per_thread);
        assert_eq!(m.get_cloned(&0), Some(1));
    }

    #[test]
    fn test_concurrent_insert_then_remove_half() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let threads = 16usize;
        let per = 2000usize;
        let barrier = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                let base = t * per;
                for i in 0..per {
                    let k = base + i;
                    m.insert(k, k);
                }
                for i in 0..per {
                    let k = base + i;
                    if k.is_multiple_of(2) {
                        m.remove(&k);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.len(), (threads * per) / 2);
        assert_eq!(m.get_cloned(&0), None);
        assert_eq!(m.get_cloned(&1), Some(1));
    }

    #[test]
    fn test_concurrent_reads_during_writes_smoke() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let running = Arc::new(AtomicUsize::new(1));

        let writer = {
            let m = Arc::clone(&m);
            let running = Arc::clone(&running);
            thread::spawn(move || {
                for i in 0..50_000usize {
                    m.insert(i % 10_000, i);
                    if i % 3 == 0 {
                        m.remove(&(i % 10_000));
                    }
                }
                running.store(0, Ordering::SeqCst);
            })
        };

        let mut readers = Vec::new();
        for _ in 0..8 {
            let m = Arc::clone(&m);
            let running = Arc::clone(&running);
            readers.push(thread::spawn(move || {
                let mut hits = 0usize;
                while running.load(Ordering::SeqCst) != 0 {
                    let k = hits % 10_000;
                    if m.contains_key(&k) {
                        let _ = m.get_cloned(&k);
                    }
                    hits = hits.wrapping_add(1);
                }
            }));
        }

        writer.join().unwrap();
        for r in readers {
            r.join().unwrap();
        }
        assert!(m.len() <= 10_000);
    }

    #[test]
    fn test_high_contention_insert_overwrite_same_key() {
        let m = Arc::new(ConcurrentHashMap::<&'static str, usize>::with_shard_count(
            64,
        ));
        let threads = 64usize;
        let barrier = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                m.insert("k", t);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let v = m.get_cloned(&"k").unwrap();
        assert!(v < threads);
    }

    #[test]
    fn test_concurrent_clear_and_insert_smoke() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        for i in 0..10_000 {
            m.insert(i, i);
        }

        let threads = 16usize;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                if t % 2 == 0 {
                    m.clear();
                } else {
                    for i in 0..2000 {
                        let k = t * 10_000 + i;
                        m.insert(k, k);
                    }
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(m.len() <= threads * 2000);
    }

    #[test]
    fn test_concurrent_put_if_absent_many_keys() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let threads = 16usize;
        let per = 5000usize;
        let barrier = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                barrier.wait();
                for i in 0..per {
                    let k = i;
                    let _ = m.put_if_absent(k, t);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.len(), per);
        for i in 0..per {
            assert!(m.get_cloned(&i).is_some());
        }
    }

    #[test]
    fn test_len_eventually_matches_after_threads_join() {
        let m = Arc::new(ConcurrentHashMap::<usize, usize>::with_shard_count(64));
        let threads = 8usize;
        let per = 3000usize;

        let mut handles = Vec::new();
        for t in 0..threads {
            let m = Arc::clone(&m);
            handles.push(thread::spawn(move || {
                let base = t * per;
                for i in 0..per {
                    m.insert(base + i, 1);
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(m.len(), threads * per);
    }

    #[test]
    fn test_get_cloned_requires_clone_semantics() {
        let m = ConcurrentHashMap::new();
        m.insert("k", vec![1, 2, 3]);
        let v = m.get_cloned(&"k").unwrap();
        assert_eq!(v, vec![1, 2, 3]);
    }

    #[test]
    fn test_compute_can_delete_missing_key() {
        let m = ConcurrentHashMap::<&'static str, usize>::new();
        let out = m.compute("k", |_| (None, 1));
        assert_eq!(out, 1);
        assert_eq!(m.get_cloned(&"k"), None);
    }
}
