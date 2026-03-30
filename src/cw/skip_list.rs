//! # 跳表（Skip List）实现
//!
//! 跳表是一种基于概率的数据结构，支持高效的查找、插入和删除操作。
//! 平均时间复杂度为 O(log n)，空间复杂度为 O(n)。
//!
//! ## 公开 API
//!
//! ### 创建跳表
//! - `SkipList::new()` → `SkipList<K, V>` - 使用默认配置创建跳表
//! - `SkipList::with_config(config)` → `SkipList<K, V>` - 使用自定义配置创建跳表
//!
//! ### 基本操作
//! - `insert(key, value)` → `Option<V>` - 插入或更新键值对，返回旧值（如果有）
//! - `get(&key)` → `Option<&V>` - 查找键对应的值
//! - `remove(&key)` → `Option<V>` - 删除指定键，返回被删除的值
//! - `contains_key(&key)` → `bool` - 检查是否包含指定键
//!
//! ### 状态查询
//! - `len()` → `usize` - 返回跳表中的元素个数
//! - `is_empty()` → `bool` - 判断跳表是否为空
//! - `level()` → `usize` - 返回当前跳表的层级数
//!
//! ### 边界查询
//! - `first()` → `Option<(&K, &V)>` - 获取最小键值对
//! - `last()` → `Option<(&K, &V)>` - 获取最大键值对
//! - `lower_bound(&key)` → `Option<(&K, &V)>` - 查找大于等于给定键的第一个元素
//! - `upper_bound(&key)` → `Option<(&K, &V)>` - 查找大于给定键的第一个元素
//! - `range(&start, &end)` → `Vec<(&K, &V)>` - 获取指定范围内的所有元素（左闭右开）
//!
//! ### 其他操作
//! - `clear()` → `()` - 清空跳表
//! - `iter()` → `SkipListIter<'_, K, V>` - 获取迭代器
//!
//! ## 使用示例
//!
//! ```rust
//! use rust_tools::cw::SkipList;
//!
//! let mut sl = SkipList::new();
//! sl.insert(1, "one");
//! sl.insert(2, "two");
//! sl.insert(3, "three");
//!
//! assert_eq!(sl.get(&2), Some(&"two"));
//! assert_eq!(sl.len(), 3);
//!
//! for (k, v) in sl.iter() {
//!     println!("{}: {}", k, v);
//! }
//! ```

use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// 跳表节点
struct Node<K, V> {
    key: K,
    value: V,
    /// 前向指针，每个元素对应一个层级的指针
    forward: Vec<Option<NonNull<Node<K, V>>>>,
}

impl<K, V> Node<K, V> {
    fn new(key: K, value: V, level: usize) -> Self {
        Node {
            key,
            value,
            forward: vec![None; level],
        }
    }
}

/// 跳表配置
#[derive(Clone, Debug)]
pub struct SkipListConfig {
    /// 最大层级
    pub max_level: usize,
    /// 节点晋升概率 (0.0 ~ 1.0)
    pub promotion_probability: f64,
}

impl Default for SkipListConfig {
    fn default() -> Self {
        SkipListConfig {
            max_level: 16,
            promotion_probability: 0.5,
        }
    }
}

/// 跳表数据结构
/// 
/// 跳表是一种基于链表的数据结构，通过多层索引实现高效的查找、插入和删除操作。
/// 平均时间复杂度：查找 O(log n)，插入 O(log n)，删除 O(log n)
/// 空间复杂度：O(n)
pub struct SkipList<K, V> {
    /// 头节点（哨兵节点）
    head: NonNull<Node<K, V>>,
    /// 当前最大层级
    level: usize,
    /// 元素数量
    length: usize,
    /// 配置
    config: SkipListConfig,
    _marker: PhantomData<Box<Node<K, V>>>,
}

impl<K, V> SkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    /// 创建一个新的跳表，使用默认配置
    pub fn new() -> Self {
        Self::with_config(SkipListConfig::default())
    }

    /// 使用指定配置创建跳表
    pub fn with_config(config: SkipListConfig) -> Self {
        // 创建哨兵节点
        // 使用 MaybeUninit 来避免 zeroed 问题
        let head = Box::new(Node {
            key: unsafe {
                // 哨兵节点的 key 永远不会被读取，所以这是安全的
                std::mem::MaybeUninit::<K>::uninit().assume_init()
            },
            value: unsafe {
                // 哨兵节点的 value 永远不会被读取，所以这是安全的
                std::mem::MaybeUninit::<V>::uninit().assume_init()
            },
            forward: vec![None; config.max_level],
        });
        
        let head_ptr = NonNull::new(Box::into_raw(head)).unwrap();
        
        SkipList {
            head: head_ptr,
            level: 1,
            length: 0,
            config,
            _marker: PhantomData,
        }
    }

    /// 获取跳表中的元素数量
    pub fn len(&self) -> usize {
        self.length
    }

    /// 判断跳表是否为空
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// 获取当前最大层级
    pub fn level(&self) -> usize {
        self.level
    }

    /// 生成随机层级
    fn random_level(&self) -> usize {
        let mut lvl = 1;
        let mut rng = rand::random::<f64>();
        
        while lvl < self.config.max_level 
            && rng < self.config.promotion_probability 
        {
            lvl += 1;
            rng = rand::random::<f64>();
        }
        
        lvl
    }

    /// 查找节点，返回更新指针和找到的节点（如果存在）
    fn find(&self, key: &K) -> (Vec<NonNull<Node<K, V>>>, Option<NonNull<Node<K, V>>>) {
        let mut update: Vec<NonNull<Node<K, V>>> = vec![self.head; self.config.max_level];
        let mut current = self.head;
        
        unsafe {
            for i in (0..self.level).rev() {
                while let Some(next) = (&(*current.as_ptr()).forward)[i] {
                    if (*next.as_ptr()).key < *key {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = current;
            }
            
            // 检查下一个节点是否是要找的节点
            if let Some(next) = (&(*current.as_ptr()).forward).get(0).and_then(|&x| x) {
                if (*next.as_ptr()).key == *key {
                    return (update, Some(next));
                }
            }
        }
        
        (update, None)
    }

    /// 查找键对应的值
    pub fn get(&self, key: &K) -> Option<&V> {
        let (_, node_ptr) = self.find(key);
        if let Some(ptr) = node_ptr {
            unsafe {
                return Some(&(*ptr.as_ptr()).value);
            }
        }
        None
    }

    /// 插入或更新键值对
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let (update, existing) = self.find(&key);
        
        unsafe {
            if let Some(node_ptr) = existing {
                // 键已存在，更新值
                let old_value = std::ptr::read(&(*node_ptr.as_ptr()).value);
                std::ptr::write(&mut (*node_ptr.as_ptr()).value, value);
                return Some(old_value);
            }
            
            // 生成新节点的层级
            let new_level = self.random_level();
            
            // 如果新层级超过当前最大层级，更新 update 数组
            let mut update = update;
            if new_level > self.level {
                for i in self.level..new_level {
                    update[i] = self.head;
                }
                self.level = new_level;
            }
            
            // 创建新节点
            let new_node = Box::new(Node::new(key, value, new_level));
            let new_node_ptr = NonNull::new(Box::into_raw(new_node)).unwrap();
            
            // 在每一层插入新节点
            for i in 0..new_level {
                (&mut (*new_node_ptr.as_ptr()).forward)[i] = (&mut (*update[i].as_ptr()).forward)[i];
                (&mut (*update[i].as_ptr()).forward)[i] = Some(new_node_ptr);
            }
            
            self.length += 1;
            None
        }
    }

    /// 删除指定键的节点
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (update, existing) = self.find(key);
        
        unsafe {
            let node_ptr = existing?;
            
            // 验证节点确实存在于跳表中
            let node_key = &(*node_ptr.as_ptr()).key;
            if node_key != key {
                return None;
            }
            
            // 从每一层中移除节点
            for i in 0..self.level {
                let current_forward = (&mut (*update[i].as_ptr()).forward)[i];
                if current_forward == Some(node_ptr) {
                    (&mut (*update[i].as_ptr()).forward)[i] = (&(*node_ptr.as_ptr()).forward)[i];
                }
            }
            
            // 读取被删除节点的值
            let removed_value = std::ptr::read(&(*node_ptr.as_ptr()).value);
            
            // 释放节点内存
            drop(Box::from_raw(node_ptr.as_ptr()));
            
            // 更新最大层级
            while self.level > 1 && (&(*self.head.as_ptr()).forward)[self.level - 1].is_none() {
                self.level -= 1;
            }
            
            self.length -= 1;
            Some(removed_value)
        }
    }

    /// 检查是否包含指定键
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// 获取第一个元素（最小键）
    pub fn first(&self) -> Option<(&K, &V)> {
        unsafe {
            if let Some(first) = (&(*self.head.as_ptr()).forward).get(0).and_then(|&x| x) {
                let node = &*first.as_ptr();
                return Some((&node.key, &node.value));
            }
        }
        None
    }

    /// 获取最后一个元素（最大键）
    pub fn last(&self) -> Option<(&K, &V)> {
        if self.is_empty() {
            return None;
        }
        
        unsafe {
            let mut current = self.head;
            for i in (0..self.level).rev() {
                while let Some(next) = (&(*current.as_ptr()).forward)[i] {
                    current = next;
                }
            }
            
            if current == self.head {
                return None;
            }
            
            let node = &*current.as_ptr();
            Some((&node.key, &node.value))
        }
    }

    /// 查找大于等于给定键的第一个元素
    pub fn lower_bound(&self, key: &K) -> Option<(&K, &V)> {
        unsafe {
            let mut current = self.head;
            for i in (0..self.level).rev() {
                while let Some(next) = (&(*current.as_ptr()).forward)[i] {
                    if (*next.as_ptr()).key < *key {
                        current = next;
                    } else {
                        break;
                    }
                }
            }
            
            if let Some(next) = (&(*current.as_ptr()).forward).get(0).and_then(|&x| x) {
                let node = &*next.as_ptr();
                return Some((&node.key, &node.value));
            }
        }
        None
    }

    /// 查找大于给定键的第一个元素
    pub fn upper_bound(&self, key: &K) -> Option<(&K, &V)> {
        unsafe {
            let mut current = self.head;
            for i in (0..self.level).rev() {
                while let Some(next) = (&(*current.as_ptr()).forward)[i] {
                    if (*next.as_ptr()).key <= *key {
                        current = next;
                    } else {
                        break;
                    }
                }
            }
            
            if let Some(next) = (&(*current.as_ptr()).forward).get(0).and_then(|&x| x) {
                let node = &*next.as_ptr();
                return Some((&node.key, &node.value));
            }
        }
        None
    }

    /// 获取指定范围内的所有元素（左闭右开区间 [start, end)）
    pub fn range(&self, start: &K, end: &K) -> Vec<(&K, &V)> {
        let mut result = Vec::new();
        
        unsafe {
            let mut current = self.head;
            for i in (0..self.level).rev() {
                while let Some(next) = (&(*current.as_ptr()).forward)[i] {
                    if (*next.as_ptr()).key < *start {
                        current = next;
                    } else {
                        break;
                    }
                }
            }
            
            // 从第一个 >= start 的节点开始遍历
            if let Some(mut next) = (&(*current.as_ptr()).forward).get(0).and_then(|&x| x) {
                while (*next.as_ptr()).key < *end {
                    let node = &*next.as_ptr();
                    result.push((&node.key, &node.value));
                    
                    if let Some(n) = (&(*next.as_ptr()).forward).get(0).and_then(|&x| x) {
                        next = n;
                    } else {
                        break;
                    }
                }
            }
        }
        
        result
    }

    /// 清空跳表
    pub fn clear(&mut self) {
        unsafe {
            let mut current = (&(*self.head.as_ptr()).forward).get(0).and_then(|&x| x);
            while let Some(node_ptr) = current {
                let next = (&(*node_ptr.as_ptr()).forward).get(0).and_then(|&x| x);
                drop(Box::from_raw(node_ptr.as_ptr()));
                current = next;
            }
            
            // 重置头节点的指针
            for i in 0..self.config.max_level {
                (&mut (*self.head.as_ptr()).forward)[i] = None;
            }
            
            self.level = 1;
            self.length = 0;
        }
    }

    /// 迭代器
    pub fn iter(&self) -> SkipListIter<'_, K, V> {
        unsafe {
            SkipListIter {
                current: (&(*self.head.as_ptr()).forward).get(0).and_then(|&x| x),
                _marker: PhantomData,
            }
        }
    }
}

impl<K, V> Drop for SkipList<K, V> {
    fn drop(&mut self) {
        unsafe {
            // 释放所有节点
            let mut current = (&(*self.head.as_ptr()).forward).get(0).and_then(|&x| x);
            while let Some(node_ptr) = current {
                let next = (&(*node_ptr.as_ptr()).forward).get(0).and_then(|&x| x);
                drop(Box::from_raw(node_ptr.as_ptr()));
                current = next;
            }
            // 释放头节点
            drop(Box::from_raw(self.head.as_ptr()));
        }
    }
}

impl<K, V> Default for SkipList<K, V>
where
    K: Ord + Clone,
    V: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> fmt::Debug for SkipList<K, V>
where
    K: Ord + Clone + fmt::Debug,
    V: Clone + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// 跳表迭代器
pub struct SkipListIter<'a, K, V> {
    current: Option<NonNull<Node<K, V>>>,
    _marker: PhantomData<&'a Node<K, V>>,
}

impl<'a, K, V> Iterator for SkipListIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        unsafe {
            if let Some(current_ptr) = self.current {
                let node = &*current_ptr.as_ptr();
                self.current = (&(*current_ptr.as_ptr()).forward).get(0).and_then(|&x| x);
                return Some((&node.key, &node.value));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut sl = SkipList::new();
        
        sl.insert(3, "three");
        sl.insert(1, "one");
        sl.insert(5, "five");
        sl.insert(2, "two");
        sl.insert(4, "four");
        
        assert_eq!(sl.get(&1), Some(&"one"));
        assert_eq!(sl.get(&2), Some(&"two"));
        assert_eq!(sl.get(&3), Some(&"three"));
        assert_eq!(sl.get(&4), Some(&"four"));
        assert_eq!(sl.get(&5), Some(&"five"));
        assert_eq!(sl.get(&6), None);
    }

    #[test]
    fn test_update() {
        let mut sl = SkipList::new();
        
        sl.insert(1, "one");
        assert_eq!(sl.get(&1), Some(&"one"));
        
        let old = sl.insert(1, "ONE");
        assert_eq!(old, Some("one"));
        assert_eq!(sl.get(&1), Some(&"ONE"));
    }

    #[test]
    fn test_remove() {
        let mut sl = SkipList::new();
        
        sl.insert(1, "one");
        sl.insert(2, "two");
        sl.insert(3, "three");
        
        assert_eq!(sl.remove(&2), Some("two"));
        assert_eq!(sl.get(&2), None);
        assert_eq!(sl.len(), 2);
        
        assert_eq!(sl.remove(&2), None);
        assert_eq!(sl.len(), 2);
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut sl = SkipList::<i32, i32>::new();
        assert!(sl.is_empty());
        assert_eq!(sl.len(), 0);
        
        sl.insert(1, 1);
        assert!(!sl.is_empty());
        assert_eq!(sl.len(), 1);
        
        sl.insert(2, 2);
        assert_eq!(sl.len(), 2);
        
        sl.remove(&1);
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn test_first_and_last() {
        let mut sl = SkipList::new();
        assert_eq!(sl.first(), None);
        assert_eq!(sl.last(), None);
        
        sl.insert(3, "three");
        sl.insert(1, "one");
        sl.insert(5, "five");
        
        assert_eq!(sl.first(), Some((&1, &"one")));
        assert_eq!(sl.last(), Some((&5, &"five")));
    }

    #[test]
    fn test_contains_key() {
        let mut sl = SkipList::new();
        sl.insert(1, "one");
        
        assert!(sl.contains_key(&1));
        assert!(!sl.contains_key(&2));
    }

    #[test]
    fn test_lower_bound() {
        let mut sl = SkipList::new();
        sl.insert(1, "one");
        sl.insert(3, "three");
        sl.insert(5, "five");
        
        assert_eq!(sl.lower_bound(&2), Some((&3, &"three")));
        assert_eq!(sl.lower_bound(&3), Some((&3, &"three")));
        assert_eq!(sl.lower_bound(&6), None);
    }

    #[test]
    fn test_upper_bound() {
        let mut sl = SkipList::new();
        sl.insert(1, "one");
        sl.insert(3, "three");
        sl.insert(5, "five");
        
        assert_eq!(sl.upper_bound(&2), Some((&3, &"three")));
        assert_eq!(sl.upper_bound(&3), Some((&5, &"five")));
        assert_eq!(sl.upper_bound(&5), None);
    }

    #[test]
    fn test_range() {
        let mut sl = SkipList::new();
        for i in 1..=10 {
            sl.insert(i, i * 10);
        }
        
        let range = sl.range(&3, &7);
        assert_eq!(range.len(), 4);
        assert_eq!(range[0], (&3, &30));
        assert_eq!(range[1], (&4, &40));
        assert_eq!(range[2], (&5, &50));
        assert_eq!(range[3], (&6, &60));
    }

    #[test]
    fn test_iter() {
        let mut sl = SkipList::new();
        sl.insert(3, "three");
        sl.insert(1, "one");
        sl.insert(2, "two");
        
        let keys: Vec<_> = sl.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3]);
    }

    #[test]
    fn test_clear() {
        let mut sl = SkipList::new();
        sl.insert(1, "one");
        sl.insert(2, "two");
        sl.insert(3, "three");
        
        assert_eq!(sl.len(), 3);
        sl.clear();
        assert_eq!(sl.len(), 0);
        assert!(sl.is_empty());
        assert_eq!(sl.get(&1), None);
    }

    #[test]
    fn test_large_dataset() {
        let mut sl = SkipList::new();
        
        // 插入大量数据
        for i in 0..1000 {
            sl.insert(i, i * 2);
        }
        
        assert_eq!(sl.len(), 1000);
        
        // 验证所有数据
        for i in 0..1000 {
            assert_eq!(sl.get(&i), Some(&(i * 2)));
        }
        
        // 删除一半数据
        for i in 0..500 {
            sl.remove(&i);
        }
        
        assert_eq!(sl.len(), 500);
        
        // 验证删除后的数据
        for i in 0..500 {
            assert_eq!(sl.get(&i), None);
        }
        for i in 500..1000 {
            assert_eq!(sl.get(&i), Some(&(i * 2)));
        }
    }

    #[test]
    fn test_debug_format() {
        let mut sl = SkipList::new();
        sl.insert(1, "one");
        sl.insert(2, "two");
        
        let debug_str = format!("{:?}", sl);
        assert!(debug_str.contains("1"));
        assert!(debug_str.contains("2"));
    }
}
