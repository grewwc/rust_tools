use std::{
    cell::RefCell,
    fmt::Display,
    hash::Hash,
    rc::{Rc, Weak},
};
use rustc_hash::FxHashMap;

struct Node<K, V>
where
    K: Hash + Eq + Default,
    V: Default + Display,
{
    key: K,
    val: V,
    prev: Option<Weak<RefCell<Node<K, V>>>>,
    next: Option<Rc<RefCell<Node<K, V>>>>,
}

impl<K, V> Node<K, V>
where
    K: Eq + Default + Hash,
    V: Default + Display,
{
    fn new(k: K, v: V) -> Self {
        Self {
            key: k,
            val: v,
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
}

impl<K, V> LruCache<K, V>
where
    K: Eq + Hash + Default + Clone,
    V: Display + Default,
{
    pub fn new(cap: usize) -> Self {
        let dummy_head = Rc::new(RefCell::new(Node::new(K::default(), V::default())));
        let dummy_tail = Rc::new(RefCell::new(Node::new(K::default(), V::default())));
        dummy_head.borrow_mut().next = Some(dummy_tail.clone());
        dummy_tail.borrow_mut().prev = Some(Rc::downgrade(&dummy_head));
        Self {
            map: FxHashMap::default(),
            head: dummy_head,
            tail: dummy_tail,
            len: 0usize,
            cap: cap,
        }
    }

    pub fn put(&mut self, k: K, v: V) {
        if self.map.contains_key(&k) {
            let node = self.map.get(&k).unwrap();
            node.borrow_mut().val = v;
            self.move_node_to_front(node.clone());
        } else {
            let new_node = Rc::new(RefCell::new(Node::new(k.clone(), v)));
            if self.len == self.cap {
                if let Some(removed) = self.remove_tail() {
                    self.map.remove(&removed.borrow().key);
                    self.len -= 1;
                }
            }
            self.add_to_front(new_node.clone());
            self.map.insert(k, new_node.clone());
            self.len += 1;
        }
    }

    pub fn get(&mut self, k: K) -> Option<&V> {
        if let Some(node) = self.map.get(&k) {
            let x = node.clone().as_ptr();
            let ret = unsafe { Some(&(*x).val) };
            self.move_node_to_front(node.clone());
            return ret;
        }
        None
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn cap(&self) -> usize {
        self.cap
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
}
