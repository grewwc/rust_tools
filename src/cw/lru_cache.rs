//! Least-recently-used (LRU) cache with optional time-to-live expiry.
//!
//! [`LruCache`] holds at most `cap` entries and evicts the least recently used
//! one when a fresh key arrives at capacity. Reading ([`get`](LruCache::get),
//! [`get_mut`](LruCache::get_mut)) or overwriting a key marks it most recently
//! used; [`peek`](LruCache::peek) and [`contains_key`](LruCache::contains_key)
//! observe without affecting recency.
//!
//! With a non-negative TTL every inserted entry carries an absolute deadline.
//! Expired entries behave as absent through every read API (`get`, `peek`,
//! `contains_key`, `len`, `is_empty`, `peek_lru`, iteration) and [`pop`] treats
//! them as absent too. Expiry is lazy: dead slots are reclaimed from the
//! least-recently-used end whenever an insert or an explicit eviction needs
//! room, so the slot count never exceeds the capacity but a mid-chain expired
//! entry may briefly linger while the eviction takes a live entry from the LRU
//! end instead. A full purge happens only when
//! [`set_ttl_ms`](LruCache::set_ttl_ms) (re-)enables expiry.
//!
//! The cache performs no locking itself; wrap it in a `Mutex` for shared use
//! (this is what the `#[lru_cache]` macro generates). Without a TTL every
//! operation is O(1); with a TTL, inserts and explicit evictions stay
//! amortized O(1) while `len`, `is_empty`, `peek_lru` and full iteration may
//! scan past expired entries.
//!
//! [`pop`]: LruCache::pop

use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::commonw::types::FastMap;

/// One storage slot. Live slots form a doubly linked list threaded through the
/// slot vector (most recently used at `head`, least at `tail`); vacant slots
/// form an intrusive free-list stack, so removal never moves memory and a
/// reinsert reuses the slot.
#[derive(Clone, Debug)]
enum Slot<K, V> {
    Live {
        /// Shared with the map entry, so lookups never clone the key and
        /// insertion needs no `K: Clone` bound.
        key: Arc<K>,
        val: V,
        expires_at: Option<Instant>,
        prev: Option<usize>,
        next: Option<usize>,
    },
    Free { next_free: Option<usize> },
}

impl<K, V> Slot<K, V> {
    /// Borrow the live parts; a free slot here means the internal chain is
    /// corrupt, which cannot happen through the public API.
    fn live_parts(&self) -> (&Arc<K>, &V, Option<Instant>, Option<usize>, Option<usize>) {
        match self {
            Slot::Live {
                key,
                val,
                expires_at,
                prev,
                next,
            } => (key, val, *expires_at, *prev, *next),
            Slot::Free { .. } => unreachable!("chain index points at a free slot"),
        }
    }

    fn live_parts_mut(
        &mut self,
    ) -> (
        &mut Arc<K>,
        &mut V,
        &mut Option<Instant>,
        &mut Option<usize>,
        &mut Option<usize>,
    ) {
        match self {
            Slot::Live {
                key,
                val,
                expires_at,
                prev,
                next,
            } => (key, val, expires_at, prev, next),
            Slot::Free { .. } => unreachable!("chain index points at a free slot"),
        }
    }
}

/// Least-recently-used cache; see the [module docs](self) for the recency and
/// expiry semantics.
#[derive(Clone, Debug)]
pub struct LruCache<K, V> {
    /// Key to slot index. `Arc<K>` borrows as `K`, so `map.get(k)` resolves
    /// without allocating a lookup key.
    map: FastMap<Arc<K>, usize>,
    slots: Vec<Slot<K, V>>,
    /// Head of the intrusive free-list stack over vacant slots.
    free_head: Option<usize>,
    /// Most-recently-used live slot; `None` exactly when the cache is empty.
    head: Option<usize>,
    /// Least-recently-used live slot; `None` exactly when the cache is empty.
    tail: Option<usize>,
    /// Live slots, including not-yet-reclaimed expired ones. Without a TTL no
    /// entry can expire, so this is the exact length; with a TTL the read APIs
    /// filter expired entries on top of it.
    len: usize,
    cap: usize,
    /// Entry TTL in milliseconds; negative disables expiry.
    ttl_ms: i64,
}

/// Outcome of the shared insert path behind [`LruCache::put`] and
/// [`LruCache::push`].
enum Upsert<K, V> {
    /// The key already held a live entry; this is the passed-in key and the
    /// replaced value.
    Updated { key: K, old: V },
    /// A fresh key was stored without displacing anything.
    Inserted,
    /// A fresh key was stored; this live entry was evicted to make room.
    Evicted { key: Arc<K>, val: V },
    /// Zero-capacity cache: nothing was stored.
    Dropped,
}

fn is_expired_at(expires_at: Option<Instant>, now: Instant) -> bool {
    matches!(expires_at, Some(t) if t <= now)
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash,
{
    pub fn new(cap: usize) -> Self {
        Self {
            map: FastMap::default(),
            slots: Vec::new(),
            free_head: None,
            head: None,
            tail: None,
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

    /// Replace the TTL. Entries inserted while expiry was disabled carry no
    /// deadline and stay live; enabling expiry purges only entries whose
    /// deadline already passed.
    pub fn set_ttl_ms(&mut self, ttl_ms: i64) {
        self.ttl_ms = ttl_ms;
        if self.ttl_ms >= 0 {
            self.purge_expired_all(Instant::now());
        }
    }

    pub fn ttl_ms(&self) -> i64 {
        self.ttl_ms
    }

    /// Insert or overwrite. Returns the old value when the key already held a
    /// live entry; an expired previous entry counts as absent and yields
    /// `None`. A zero-capacity cache stores nothing and returns `None`.
    pub fn put(&mut self, k: K, v: V) -> Option<V> {
        let now = Instant::now();
        match self.upsert(k, v, now) {
            Upsert::Updated { old, .. } => Some(old),
            Upsert::Inserted | Upsert::Evicted { .. } | Upsert::Dropped => None,
        }
    }

    /// Insert like [`put`](LruCache::put), but also report the displaced
    /// entry: on a key update the passed-in key paired with the old value, on
    /// a capacity eviction the evicted pair, otherwise `None`.
    pub fn push(&mut self, k: K, v: V) -> Option<(K, V)> {
        let now = Instant::now();
        match self.upsert(k, v, now) {
            Upsert::Updated { key, old } => Some((key, old)),
            Upsert::Evicted { key, val } => Some((Self::unkey(key), val)),
            Upsert::Inserted | Upsert::Dropped => None,
        }
    }

    /// Look up an owned key, marking it most recently used. Returns a clone of
    /// the value, or `None` when the key is absent or expired. Prefer
    /// [`get_ref`](LruCache::get_ref) when the key is only borrowed.
    pub fn get(&mut self, k: K) -> Option<V>
    where
        V: Clone,
    {
        self.get_ref(&k)
    }

    /// Look up a borrowed key, marking it most recently used. Returns a clone
    /// of the value, or `None` when the key is absent or expired.
    pub fn get_ref(&mut self, k: &K) -> Option<V>
    where
        V: Clone,
    {
        let now = Instant::now();
        let idx = self.map.get(k).copied()?;
        if is_expired_at(self.slots[idx].live_parts().2, now) {
            self.remove_slot(idx);
            return None;
        }
        let val = self.slots[idx].live_parts().1.clone();
        self.move_to_front(idx);
        Some(val)
    }

    /// Look up a borrowed key without affecting recency and without cloning
    /// the value. Returns `None` when the key is absent or expired.
    pub fn peek(&self, k: &K) -> Option<&V> {
        let idx = self.map.get(k).copied()?;
        let (_, val, expires_at, _, _) = self.slots[idx].live_parts();
        if is_expired_at(expires_at, Instant::now()) {
            return None;
        }
        Some(val)
    }

    /// Look up a borrowed key mutably, marking it most recently used. Returns
    /// `None` when the key is absent or expired.
    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        let now = Instant::now();
        let idx = self.map.get(k).copied()?;
        if is_expired_at(self.slots[idx].live_parts().2, now) {
            self.remove_slot(idx);
            return None;
        }
        self.move_to_front(idx);
        Some(self.slots[idx].live_parts_mut().1)
    }

    /// Remove a key, returning its value. An expired entry counts as absent:
    /// it is dropped and `None` is returned.
    pub fn pop(&mut self, k: &K) -> Option<V> {
        let now = Instant::now();
        let idx = self.map.get(k).copied()?;
        let expired = is_expired_at(self.slots[idx].live_parts().2, now);
        let (_, val) = self.remove_slot(idx);
        if expired { None } else { Some(val) }
    }

    /// Remove and return the least-recently-used live entry, reclaiming
    /// expired entries from the LRU end first. Returns `None` when no live
    /// entry remains.
    pub fn pop_lru(&mut self) -> Option<(K, V)> {
        let now = Instant::now();
        self.reclaim_expired_tail(now);
        let tail = self.tail?;
        let (key, val) = self.remove_slot(tail);
        Some((Self::unkey(key), val))
    }

    /// Borrow the least-recently-used live entry without removing it, skipping
    /// expired entries. Takes `&self`, so unlike [`pop_lru`](LruCache::pop_lru)
    /// it cannot reclaim the entries it skips; with a TTL this may scan.
    pub fn peek_lru(&self) -> Option<(&K, &V)> {
        let now = Instant::now();
        let mut cur = self.tail;
        while let Some(idx) = cur {
            let (key, val, expires_at, prev, _) = self.slots[idx].live_parts();
            if !is_expired_at(expires_at, now) {
                return Some((key, val));
            }
            cur = prev;
        }
        None
    }

    /// Iterate over live entries from most to least recently used, skipping
    /// expired entries. Expiry is judged against the instant the iterator was
    /// created.
    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            slots: &self.slots,
            next: self.head,
            now: Instant::now(),
        }
    }

    /// Number of live entries. Without a TTL this is O(1); with a TTL it
    /// walks the chain once to skip expired entries.
    pub fn len(&self) -> usize {
        if self.ttl_ms < 0 {
            return self.len;
        }
        let now = Instant::now();
        let mut count = 0;
        let mut cur = self.head;
        while let Some(idx) = cur {
            let (_, _, expires_at, _, next) = self.slots[idx].live_parts();
            if !is_expired_at(expires_at, now) {
                count += 1;
            }
            cur = next;
        }
        count
    }

    pub fn is_empty(&self) -> bool {
        if self.ttl_ms < 0 {
            return self.len == 0;
        }
        // Early exit on the first live entry instead of counting all of them.
        let now = Instant::now();
        let mut cur = self.head;
        while let Some(idx) = cur {
            let (_, _, expires_at, _, next) = self.slots[idx].live_parts();
            if !is_expired_at(expires_at, now) {
                return false;
            }
            cur = next;
        }
        true
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Set the capacity, evicting least-recently-used entries (after
    /// reclaiming expired ones from the LRU end) until the cache fits.
    /// Shrinking to zero empties the cache; growing keeps every entry.
    pub fn resize(&mut self, cap: usize) {
        self.cap = cap;
        self.reclaim_expired_tail(Instant::now());
        while self.len > self.cap {
            let tail = self
                .tail
                .expect("a non-empty cache holds a least-recently-used entry");
            self.remove_slot(tail);
        }
    }

    /// True when the key holds a live entry. Never affects recency and never
    /// reclaims; an expired entry simply reports `false`.
    pub fn contains_key(&self, k: &K) -> bool {
        let Some(&idx) = self.map.get(k) else {
            return false;
        };
        if self.ttl_ms < 0 {
            return true;
        }
        !is_expired_at(self.slots[idx].live_parts().2, Instant::now())
    }

    /// Drop every entry, releasing the slots. Capacity and TTL are kept.
    pub fn clear(&mut self) {
        self.map.clear();
        self.slots.clear();
        self.free_head = None;
        self.head = None;
        self.tail = None;
        self.len = 0;
    }

    fn calc_expires_at(&self, now: Instant) -> Option<Instant> {
        if self.ttl_ms < 0 {
            return None;
        }
        now.checked_add(Duration::from_millis(self.ttl_ms as u64))
    }

    /// Unwrap a key whose map and slot entries were both removed: those were
    /// the only two clones, so exactly one remains.
    fn unkey(key: Arc<K>) -> K {
        match Arc::try_unwrap(key) {
            Ok(k) => k,
            Err(_) => unreachable!("evicted keys have no outstanding clones"),
        }
    }

    /// Shared insert path: update the live entry, or store a fresh key after
    /// making room (expired entries at the LRU end first, then one live
    /// eviction when still full).
    fn upsert(&mut self, k: K, v: V, now: Instant) -> Upsert<K, V> {
        if let Some(idx) = self.map.get(&k).copied() {
            if is_expired_at(self.slots[idx].live_parts().2, now) {
                self.remove_slot(idx);
            } else {
                let expires_at = self.calc_expires_at(now);
                let old = {
                    let (_, val, slot_expires_at, _, _) = self.slots[idx].live_parts_mut();
                    let old = std::mem::replace(val, v);
                    *slot_expires_at = expires_at;
                    old
                };
                self.move_to_front(idx);
                return Upsert::Updated { key: k, old };
            }
        }
        if self.cap == 0 {
            return Upsert::Dropped;
        }
        self.reclaim_expired_tail(now);
        let evicted = if self.len == self.cap {
            let tail = self
                .tail
                .expect("a full cache holds a least-recently-used entry");
            let (key, val) = self.remove_slot(tail);
            Some((key, val))
        } else {
            None
        };
        self.insert_fresh(Arc::new(k), v, self.calc_expires_at(now));
        match evicted {
            Some((key, val)) => Upsert::Evicted { key, val },
            None => Upsert::Inserted,
        }
    }

    /// Store a key known to be absent (or just removed), reusing a vacant slot
    /// when one exists. The caller must have ensured capacity.
    fn insert_fresh(&mut self, key: Arc<K>, val: V, expires_at: Option<Instant>) {
        let idx = if let Some(idx) = self.free_head {
            let next = match &self.slots[idx] {
                Slot::Free { next_free } => *next_free,
                Slot::Live { .. } => unreachable!("free list points at a live slot"),
            };
            self.free_head = next;
            idx
        } else {
            let idx = self.slots.len();
            self.slots.push(Slot::Free { next_free: None });
            idx
        };
        self.slots[idx] = Slot::Live {
            key: key.clone(),
            val,
            expires_at,
            prev: None,
            next: None,
        };
        self.map.insert(key, idx);
        self.push_front(idx);
        self.len += 1;
    }

    /// Unlink a live slot from the chain and recycle it, returning its key and
    /// value. Drops both key clones (map and slot) on the way out.
    fn remove_slot(&mut self, idx: usize) -> (Arc<K>, V) {
        self.detach(idx);
        let slot = std::mem::replace(
            &mut self.slots[idx],
            Slot::Free {
                next_free: self.free_head,
            },
        );
        self.free_head = Some(idx);
        match slot {
            Slot::Live { key, val, .. } => {
                self.map.remove(&key);
                self.len -= 1;
                (key, val)
            }
            Slot::Free { .. } => unreachable!("double free of a cache slot"),
        }
    }

    /// Drop expired entries from the LRU end. Each slot is reclaimed at most
    /// once, so this is amortized O(1) per call; a no-op without a TTL.
    fn reclaim_expired_tail(&mut self, now: Instant) {
        if self.ttl_ms < 0 {
            return;
        }
        while let Some(tail) = self.tail {
            if !is_expired_at(self.slots[tail].live_parts().2, now) {
                break;
            }
            self.remove_slot(tail);
        }
    }

    /// Drop every expired entry. Only [`set_ttl_ms`](LruCache::set_ttl_ms) pays
    /// for a full walk; the hot paths reclaim lazily from the LRU end.
    fn purge_expired_all(&mut self, now: Instant) {
        if self.ttl_ms < 0 {
            return;
        }
        let mut cur = self.head;
        while let Some(idx) = cur {
            let (_, _, expires_at, _, next) = self.slots[idx].live_parts();
            if is_expired_at(expires_at, now) {
                // Detaching preserves the prefetched successor.
                self.remove_slot(idx);
            }
            cur = next;
        }
    }

    fn move_to_front(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return;
        }
        self.detach(idx);
        self.push_front(idx);
    }

    fn push_front(&mut self, idx: usize) {
        let head = self.head;
        let (_, _, _, prev, next) = self.slots[idx].live_parts_mut();
        *prev = None;
        *next = head;
        if let Some(h) = head {
            *self.slots[h].live_parts_mut().3 = Some(idx);
        } else {
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    fn detach(&mut self, idx: usize) {
        let (_, _, _, prev, next) = self.slots[idx].live_parts();
        if let Some(p) = prev {
            *self.slots[p].live_parts_mut().4 = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            *self.slots[n].live_parts_mut().3 = prev;
        } else {
            self.tail = prev;
        }
    }
}

/// Live entries from most to least recently used; see
/// [`LruCache::iter`].
#[derive(Clone, Debug)]
pub struct Iter<'a, K, V> {
    slots: &'a [Slot<K, V>],
    next: Option<usize>,
    now: Instant,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(idx) = self.next {
            match &self.slots[idx] {
                Slot::Live {
                    key,
                    val,
                    expires_at,
                    next,
                    ..
                } => {
                    self.next = *next;
                    if !is_expired_at(*expires_at, self.now) {
                        return Some((key, val));
                    }
                }
                // Unreachable by construction (the chain holds live slots
                // only); stop gracefully rather than panic inside an iterator.
                Slot::Free { .. } => {
                    self.next = None;
                    return None;
                }
            }
        }
        None
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
    }

    #[test]
    fn put_returns_old_value_and_refreshes_recency() {
        let mut c = LruCache::new(2);
        assert_eq!(c.put(1, 10), None);
        c.put(2, 20);
        assert_eq!(c.put(1, 11), Some(10));
        // The update marked 1 most recently used, so 2 is evicted next.
        c.put(3, 30);
        assert_eq!(c.get(1), Some(11));
        assert_eq!(c.get(2), None);
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn update_at_capacity_does_not_evict() {
        let mut c = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.put(1, 11), Some(10));
        assert_eq!(c.len(), 2);
        assert!(c.contains_key(&1));
        assert!(c.contains_key(&2));
    }

    #[test]
    fn zero_capacity_stores_nothing() {
        let mut c: LruCache<i32, i32> = LruCache::new(0);
        assert_eq!(c.put(1, 10), None);
        assert_eq!(c.push(1, 10), None);
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.get_ref(&1), None);
        assert_eq!(c.pop_lru(), None);
    }

    #[test]
    fn pop_removes_entry() {
        let mut c = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.pop(&1), Some(10));
        assert_eq!(c.pop(&1), None);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get_ref(&2), Some(20));
        // The freed slot is reused instead of growing the table.
        c.put(3, 30);
        assert_eq!(c.len(), 2);
        assert!(c.slots.len() <= 2);
    }

    #[test]
    fn push_reports_displaced_entry() {
        let mut c = LruCache::new(2);
        assert_eq!(c.push(1, 10), None);
        c.push(2, 20);
        // Update: the passed-in key paired with the old value.
        assert_eq!(c.push(1, 11), Some((1, 10)));
        // Fresh key at capacity: the evicted least-recently-used pair.
        assert_eq!(c.push(3, 30), Some((2, 20)));
        assert_eq!(c.get_ref(&1), Some(11));
        assert_eq!(c.get_ref(&3), Some(30));
    }

    #[test]
    fn peek_observes_without_touching_recency() {
        let mut c = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        assert_eq!(c.peek(&1), Some(&10));
        assert_eq!(c.peek(&9), None);
        // The peek did not save 1 from eviction.
        c.put(3, 30);
        assert_eq!(c.get_ref(&1), None);
        assert_eq!(c.get_ref(&2), Some(20));
    }

    #[test]
    fn get_mut_updates_value_and_recency() {
        let mut c = LruCache::new(2);
        c.put(1, 10);
        c.put(2, 20);
        *c.get_mut(&1).unwrap() += 1;
        assert_eq!(c.get_mut(&9), None);
        // The mutable access marked 1 most recently used.
        c.put(3, 30);
        assert_eq!(c.get_ref(&1), Some(11));
        assert_eq!(c.get_ref(&2), None);
    }

    #[test]
    fn pop_lru_and_peek_lru_follow_recency() {
        let mut c = LruCache::new(3);
        assert_eq!(c.pop_lru(), None);
        assert_eq!(c.peek_lru(), None);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        assert_eq!(c.peek_lru(), Some((&1, &10)));
        c.get_ref(&1);
        assert_eq!(c.peek_lru(), Some((&2, &20)));
        assert_eq!(c.pop_lru(), Some((2, 20)));
        assert_eq!(c.len(), 2);
        // peek_lru does not remove.
        assert_eq!(c.peek_lru(), Some((&3, &30)));
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn resize_evicts_from_lru_end() {
        let mut c = LruCache::new(3);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        c.resize(2);
        assert_eq!(c.cap(), 2);
        assert_eq!(c.len(), 2);
        assert!(!c.contains_key(&1));
        c.resize(5);
        assert_eq!(c.len(), 2);
        c.put(4, 40);
        c.put(5, 50);
        assert_eq!(c.len(), 4);
        c.resize(0);
        assert!(c.is_empty());
        assert_eq!(c.cap(), 0);
    }

    #[test]
    fn iter_yields_mru_to_lru() {
        let mut c = LruCache::new(3);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        c.get_ref(&1);
        let order: Vec<(i32, i32)> = c.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(order, vec![(1, 10), (3, 30), (2, 20)]);
    }

    #[test]
    fn iter_skips_expired_entries() {
        let mut c = LruCache::with_ttl(3, 20);
        c.put(1, 10);
        sleep(Duration::from_millis(50));
        c.put(2, 20);
        c.put(3, 30);
        let order: Vec<(i32, i32)> = c.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(order, vec![(3, 30), (2, 20)]);
    }

    #[test]
    fn expired_entries_are_invisible_and_reclaimed() {
        let mut c = LruCache::with_ttl(2, 20);
        c.put(1, 10);
        c.put(2, 20);
        sleep(Duration::from_millis(50));
        assert_eq!(c.len(), 0);
        assert!(c.is_empty());
        assert_eq!(c.get_ref(&1), None);
        assert_eq!(c.peek(&1), None);
        assert!(!c.contains_key(&1));
        assert_eq!(c.peek_lru(), None);
        assert_eq!(c.pop(&2), None);
        assert_eq!(c.pop_lru(), None);
        assert_eq!(c.iter().count(), 0);
        // The dead slots are reclaimed, so fresh keys fit without eviction.
        c.put(3, 30);
        c.put(4, 40);
        assert_eq!(c.len(), 2);
        assert_eq!(c.get_ref(&3), Some(30));
        assert_eq!(c.get_ref(&4), Some(40));
    }

    #[test]
    fn expired_put_reinserts_without_growing() {
        let mut c = LruCache::with_ttl(2, 20);
        c.put(1, 10);
        c.put(2, 20);
        sleep(Duration::from_millis(50));
        // The expired value counts as absent, so no old value is reported.
        assert_eq!(c.put(1, 11), None);
        assert_eq!(c.len(), 1);
        assert_eq!(c.get_ref(&1), Some(11));
    }

    #[test]
    fn enabling_ttl_later_does_not_retro_expire() {
        let mut c: LruCache<i32, i32> = LruCache::new(2);
        c.put(1, 10);
        c.set_ttl_ms(20);
        sleep(Duration::from_millis(50));
        // The entry never received a deadline, so it stays live.
        assert_eq!(c.get_ref(&1), Some(10));
        // But entries inserted after enabling do expire.
        c.put(2, 20);
        sleep(Duration::from_millis(50));
        assert_eq!(c.get_ref(&2), None);
        assert_eq!(c.get_ref(&1), Some(10));
    }

    #[test]
    fn churn_reuses_slots_without_growth() {
        let mut c = LruCache::new(2);
        for i in 0..100 {
            c.put(i % 2, i);
            c.pop(&(i % 2));
        }
        assert!(c.is_empty());
        assert!(c.slots.len() <= 2);
        c.put(1, 10);
        c.put(2, 20);
        c.put(3, 30);
        assert_eq!(c.len(), 2);
        assert!(c.slots.len() <= 3);
    }

    #[test]
    fn clear_empties_and_keeps_capacity() {
        let mut c = LruCache::with_ttl(2, 1000);
        c.put(1, 10);
        c.put(2, 20);
        c.clear();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
        assert_eq!(c.cap(), 2);
        assert_eq!(c.ttl_ms(), 1000);
        c.put(3, 30);
        assert_eq!(c.get_ref(&3), Some(30));
    }
}
