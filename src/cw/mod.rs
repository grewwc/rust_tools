//! # 容器和数据结构 (Container Tools)
//!
//! 本模块提供多种常用的数据结构和容器实现，包括队列、栈、树、图、哈希表等。
//!
//! ## 数据结构列表
//!
//! ### 基础数据结构
//!
//! - [`queue::Queue`] - 先进先出队列
//! - [`stack::Stack`] - 后进先出栈
//! - [`deque_list::DequeList`] - 双端队列
//!
//! ### 映射和集合
//!
//! - [`ordered_map::OrderedMap`] - 保持插入顺序的映射
//! - [`ordered_set::OrderedSet`] - 保持插入顺序的集合
//! - [`tree_map::TreeMap`] - 基于树的有序映射
//! - [`tree_set::TreeSet`] - 基于树的有序集合
//! - [`concurrent_hash_map::ConcurrentHashMap`] - 并发安全的哈希映射
//!
//! ### 高级数据结构
//!
//! - [`lru_cache::LruCache`] - LRU（最近最少使用）缓存
//! - [`priority_queue::MaxPriorityQueue`] / [`priority_queue::MinPriorityQueue`] - 优先队列
//! - [`bloom_filter::BloomFilter`] - 布隆过滤器
//! - [`counter::Counter`] - 计数器（元素频率统计）
//! - [`zset::ZSet`] - 有序集合（带分数排序）
//! - [`uf::UF`] - 并查集（Union-Find）
//! - [`trie::Trie`] - 前缀树（字典树）
//! - [`skip_list::SkipMap`] / [`skip_list::SkipSet`] - 跳表
//! - [`rb_tree::RbTree`] - 红黑树（API 风格）
//!
//! ### 图结构
//!
//! - [`graph::DirectedGraph`] / [`graph::UndirectedGraph`] - 有向图/无向图
//! - [`graph::WeightedDirectedGraph`] / [`graph::WeightedUndirectedGraph`] - 带权图
//! - [`graph::Edge`] - 边
//! - [`graph::Mst`] - 最小生成树
//!
//! ## 使用示例
//!
//! ### 队列
//!
//! ```rust
//! use rust_tools::cw::Queue;
//!
//! let mut queue: Queue<i32> = Queue::new();
//! queue.enqueue(1);
//! queue.enqueue(2);
//! queue.enqueue(3);
//!
//! assert_eq!(queue.dequeue(), Some(1));
//! assert_eq!(queue.dequeue(), Some(2));
//! ```
//!
//! ### LRU 缓存
//!
//! ```rust
//! use rust_tools::cw::LruCache;
//!
//! let mut cache = LruCache::new(2); // 容量为 2
//! cache.put(1, "a");
//! cache.put(2, "b");
//! cache.get(1); // 访问键 1，使其成为最近使用的
//! cache.put(3, "c"); // 键 2 会被淘汰
//!
//! assert!(cache.get(2).is_none());
//! assert_eq!(cache.get(1), Some(&"a"));
//! ```
//!
//! ### 布隆过滤器
//!
//! ```rust
//! use rust_tools::cw::BloomFilter;
//!
//! let mut bf = BloomFilter::new(1000, 3); // 1000 位，3 个哈希函数
//! bf.insert(&"hello".to_string());
//! bf.insert(&"world".to_string());
//!
//! assert!(bf.contains(&"hello".to_string()));
//! assert!(bf.contains(&"world".to_string()));
//! assert!(!bf.contains(&"rust".to_string())); // 可能为 false positive
//! ```
//!
//! ### 计数器
//!
//! ```rust
//! use rust_tools::cw::Counter;
//!
//! let mut counter: Counter<char> = Counter::new();
//! for c in "hello world".chars() {
//!     counter.inc(c);
//! }
//!
//! assert_eq!(counter.get(&'l'), 3);
//! assert_eq!(counter.get(&'o'), 2);
//! ```
//!
//! ### 有序集合
//!
//! ```rust
//! use rust_tools::cw::OrderedSet;
//!
//! let mut set = OrderedSet::new();
//! set.insert(3);
//! set.insert(1);
//! set.insert(2);
//!
//! // 保持插入顺序
//! let vec: Vec<_> = set.iter().collect();
//! assert_eq!(vec, [&3, &1, &2]);
//! ```

pub mod bloom_filter;
pub mod concurrent_hash_map;
pub mod counter;
pub mod deque_list;
pub mod graph;
pub mod lru_cache;
pub mod ordered_map;
pub mod ordered_set;
pub mod priority_queue;
pub mod queue;
pub mod stack;
pub mod tree_map;
pub mod tree_set;
pub mod rb_tree;
pub mod skip_list;
pub mod trie;
pub mod uf;
pub mod zset;

// 重新导出常用类型
pub use bloom_filter::BloomFilter;
pub use counter::Counter;
pub use deque_list::DequeList;
pub use graph::{DirectedGraph, Edge, Mst, UndirectedGraph, WeightedDirectedGraph, WeightedUndirectedGraph};
pub use lru_cache::LruCache;
pub use ordered_map::OrderedMap;
pub use ordered_set::OrderedSet;
pub use priority_queue::{MaxPriorityQueue, MinPriorityQueue};
pub use queue::Queue;
pub use skip_list::{SkipMap, SkipSet};
pub use stack::Stack;
pub use tree_map::TreeMap;
pub use tree_set::TreeSet;
pub use trie::Trie;
pub use uf::UF;
pub use zset::ZSet;
