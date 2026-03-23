use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub struct MaxPriorityQueue<T>
where
    T: Ord,
{
    data: BinaryHeap<T>,
}

impl<T> MaxPriorityQueue<T>
where
    T: Ord,
{
    pub fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: BinaryHeap::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, v: T) {
        self.data.push(v);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.peek()
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    pub fn pop_all(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len());
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }
}

impl<T> Default for MaxPriorityQueue<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

pub struct MinPriorityQueue<T>
where
    T: Ord,
{
    data: BinaryHeap<Reverse<T>>,
}

impl<T> MinPriorityQueue<T>
where
    T: Ord,
{
    pub fn new() -> Self {
        Self {
            data: BinaryHeap::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: BinaryHeap::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn push(&mut self, v: T) {
        self.data.push(Reverse(v));
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop().map(|x| x.0)
    }

    pub fn peek(&self) -> Option<&T> {
        self.data.peek().map(|x| &x.0)
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter.into_iter().map(Reverse));
    }

    pub fn pop_all(&mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.len());
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }
}

impl<T> Default for MinPriorityQueue<T>
where
    T: Ord,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{MaxPriorityQueue, MinPriorityQueue};

    #[test]
    fn test_max_priority_queue() {
        let mut q = MaxPriorityQueue::new();
        q.extend([3, 1, 2]);
        assert_eq!(q.peek(), Some(&3));
        assert_eq!(q.pop_all(), vec![3, 2, 1]);
    }

    #[test]
    fn test_min_priority_queue() {
        let mut q = MinPriorityQueue::new();
        q.extend([3, 1, 2]);
        assert_eq!(q.peek(), Some(&1));
        assert_eq!(q.pop_all(), vec![1, 2, 3]);
    }
}
