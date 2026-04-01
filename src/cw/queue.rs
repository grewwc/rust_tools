//! 队列（Queue）实现
//!
//! 基于 `LinkedList` 实现的先进先出（FIFO）队列。

use std::collections::LinkedList;

/// 先进先出队列（First-In-First-Out Queue）
///
/// 基于 `LinkedList` 实现，支持在队尾入队、队头出队的操作。
/// 所有操作的时间复杂度均为 O(1)。
///
/// # 类型参数
///
/// * `T` - 队列中存储的元素类型
///
/// # 示例
///
/// ```rust
/// use rust_tools::cw::Queue;
///
/// let mut queue: Queue<i32> = Queue::new();
///
/// // 入队
/// queue.enqueue(1);
/// queue.enqueue(2);
/// queue.enqueue(3);
///
/// // 查看队首和队尾元素
/// assert_eq!(queue.front(), Some(&1));
/// assert_eq!(queue.back(), Some(&3));
///
/// // 出队
/// assert_eq!(queue.dequeue(), Some(1));
/// assert_eq!(queue.dequeue(), Some(2));
/// assert_eq!(queue.dequeue(), Some(3));
/// assert_eq!(queue.dequeue(), None);
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：
///   - `enqueue`: O(1)
///   - `dequeue`: O(1)
///   - `front`/`back`: O(1)
///   - `len`/`is_empty`: O(1)
/// - 空间复杂度：O(n)
pub struct Queue<T> {
    data: LinkedList<T>,
}

impl<T> Queue<T> {
    /// 创建一个新的空队列
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let queue: Queue<i32> = Queue::new();
    /// assert!(queue.is_empty());
    /// ```
    pub fn new() -> Self {
        Queue {
            data: LinkedList::new(),
        }
    }
}

impl<T> Default for Queue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Queue<T> {
    /// 将元素添加到队尾（入队）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// assert_eq!(queue.len(), 2);
    /// ```
    pub fn enqueue(&mut self, val: T) {
        self.data.push_back(val);
    }

    /// 从队头移除并返回元素（出队）
    ///
    /// 如果队列为空，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// assert_eq!(queue.dequeue(), Some(1));
    /// assert_eq!(queue.dequeue(), None);
    /// ```
    pub fn dequeue(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    /// 清空队列
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// queue.clear();
    /// assert!(queue.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// 返回队列中的元素数量
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// assert_eq!(queue.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 返回队列中的元素数量（别名方法）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// assert_eq!(queue.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.size()
    }

    /// 检查队列是否为空
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// assert!(queue.is_empty());
    /// queue.enqueue(1);
    /// assert!(!queue.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 返回队首元素的引用（不移除）
    ///
    /// 如果队列为空，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// assert_eq!(queue.front(), Some(&1));
    /// ```
    pub fn front(&self) -> Option<&T> {
        self.data.front()
    }

    /// 返回队尾元素的引用（不移除）
    ///
    /// 如果队列为空，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// assert_eq!(queue.back(), Some(&2));
    /// ```
    pub fn back(&self) -> Option<&T> {
        self.data.back()
    }

    /// 返回队列的迭代器
    ///
    /// 迭代顺序为从队首到队尾。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// queue.enqueue(3);
    ///
    /// let vec: Vec<_> = queue.iter().collect();
    /// assert_eq!(vec, [&1, &2, &3]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// 从迭代器扩展队列
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.extend([1, 2, 3]);
    /// assert_eq!(queue.len(), 3);
    /// ```
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    /// 将队列转换为 `Vec`
    ///
    /// 此操作会消耗队列。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let mut queue = Queue::new();
    /// queue.enqueue(1);
    /// queue.enqueue(2);
    /// queue.enqueue(3);
    ///
    /// let vec = queue.into_vec();
    /// assert_eq!(vec, vec![1, 2, 3]);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_iter().collect()
    }
}

impl<T> From<Vec<T>> for Queue<T> {
    /// 从 `Vec` 创建队列
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::Queue;
    ///
    /// let vec = vec![1, 2, 3];
    /// let queue: Queue<i32> = Queue::from(vec);
    /// assert_eq!(queue.len(), 3);
    /// assert_eq!(queue.front(), Some(&1));
    /// ```
    fn from(v: Vec<T>) -> Self {
        Self {
            data: v.into_iter().collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Queue;

    #[test]
    fn test_queue_basic() {
        let mut q = Queue::new();
        assert!(q.is_empty());
        q.enqueue(1);
        q.enqueue(2);
        assert_eq!(q.front(), Some(&1));
        assert_eq!(q.back(), Some(&2));
        assert_eq!(q.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.dequeue(), None);
    }

    #[test]
    fn test_queue_from_vec() {
        let vec = vec![1, 2, 3];
        let mut q: Queue<i32> = Queue::from(vec);
        assert_eq!(q.dequeue(), Some(1));
        assert_eq!(q.dequeue(), Some(2));
        assert_eq!(q.dequeue(), Some(3));
    }

    #[test]
    fn test_queue_clear() {
        let mut q = Queue::new();
        q.enqueue(1);
        q.enqueue(2);
        q.clear();
        assert!(q.is_empty());
    }
}
