//! 栈（Stack）实现
//!
//! 基于 `LinkedList` 实现的后进先出（LIFO）栈。

/// 后进先出栈（Last-In-First-Out Stack）
///
/// 基于 `LinkedList` 实现，支持在栈顶压入和弹出的操作。
/// 所有操作的时间复杂度均为 O(1)。
///
/// # 类型参数
///
/// * `T` - 栈中存储的元素类型
///
/// # 示例
///
/// ```rust
/// use rust_tools::cw::stack::Stack;
///
/// let mut stack: Stack<i32> = Stack::new();
///
/// // 压栈
/// stack.push(1);
/// stack.push(2);
/// stack.push(3);
///
/// // 查看栈顶元素
/// assert_eq!(stack.top(), Some(&3));
///
/// // 弹栈
/// assert_eq!(stack.pop(), Some(3));
/// assert_eq!(stack.pop(), Some(2));
/// assert_eq!(stack.pop(), Some(1));
/// assert_eq!(stack.pop(), None);
/// ```
///
/// # 性能特征
///
/// - 时间复杂度：
///   - `push`: O(1)
///   - `pop`: O(1)
///   - `top`: O(1)
///   - `len`/`is_empty`: O(1)
/// - 空间复杂度：O(n)
pub struct Stack<T> {
    data: std::collections::LinkedList<T>,
}

impl<T> Stack<T> {
    /// 创建一个新的空栈
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let stack: Stack<i32> = Stack::new();
    /// assert!(stack.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            data: std::collections::LinkedList::new(),
        }
    }
}

impl<T> Default for Stack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Stack<T> {
    /// 将元素压入栈顶
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// assert_eq!(stack.len(), 2);
    /// ```
    pub fn push(&mut self, val: T) {
        self.data.push_back(val);
    }

    /// 从栈顶移除并返回元素
    ///
    /// 如果栈为空，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// assert_eq!(stack.pop(), Some(1));
    /// assert_eq!(stack.pop(), None);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        self.data.pop_back()
    }

    /// 清空栈
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// stack.clear();
    /// assert!(stack.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// 返回栈中的元素数量
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// assert_eq!(stack.size(), 2);
    /// ```
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// 返回栈中的元素数量（别名方法）
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// assert_eq!(stack.len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.size()
    }

    /// 检查栈是否为空
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// assert!(stack.is_empty());
    /// stack.push(1);
    /// assert!(!stack.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// 返回栈顶元素的引用（不移除）
    ///
    /// 如果栈为空，返回 `None`。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// assert_eq!(stack.top(), Some(&2));
    /// ```
    pub fn top(&self) -> Option<&T> {
        self.data.back()
    }

    /// 返回栈的迭代器
    ///
    /// 迭代顺序为从栈底到栈顶。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// stack.push(3);
    ///
    /// let vec: Vec<_> = stack.iter().collect();
    /// assert_eq!(vec, [&1, &2, &3]);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// 从迭代器扩展栈
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.extend([1, 2, 3]);
    /// assert_eq!(stack.len(), 3);
    /// ```
    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    /// 将栈转换为 `Vec`
    ///
    /// 此操作会消耗栈。迭代顺序为从栈底到栈顶。
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let mut stack = Stack::new();
    /// stack.push(1);
    /// stack.push(2);
    /// stack.push(3);
    ///
    /// let vec = stack.into_vec();
    /// assert_eq!(vec, vec![1, 2, 3]);
    /// ```
    pub fn into_vec(self) -> Vec<T> {
        self.data.into_iter().collect()
    }
}

impl<T> From<Vec<T>> for Stack<T> {
    /// 从 `Vec` 创建栈
    ///
    /// # 示例
    ///
    /// ```rust
    /// use rust_tools::cw::stack::Stack;
    ///
    /// let vec = vec![1, 2, 3];
    /// let stack: Stack<i32> = Stack::from(vec);
    /// assert_eq!(stack.len(), 3);
    /// assert_eq!(stack.top(), Some(&3));
    /// ```
    fn from(v: Vec<T>) -> Self {
        let mut s = Stack::new();
        s.extend(v);
        s
    }
}

#[cfg(test)]
mod tests {
    use super::Stack;

    #[test]
    fn test_stack_basic() {
        let mut s = Stack::new();
        assert!(s.is_empty());
        s.push(1);
        s.push(2);
        assert_eq!(s.top(), Some(&2));
        assert_eq!(s.iter().copied().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
        assert_eq!(s.pop(), None);
    }

    #[test]
    fn test_stack_from_vec() {
        let vec = vec![1, 2, 3];
        let mut s: Stack<i32> = Stack::from(vec);
        assert_eq!(s.pop(), Some(3));
        assert_eq!(s.pop(), Some(2));
        assert_eq!(s.pop(), Some(1));
    }

    #[test]
    fn test_stack_clear() {
        let mut s = Stack::new();
        s.push(1);
        s.push(2);
        s.clear();
        assert!(s.is_empty());
    }
}
