use std::collections::LinkedList;

pub struct Queue<T> {
    data: LinkedList<T>,
}

impl<T> Queue<T> {
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
    pub fn enqueue(&mut self, val: T) {
        self.data.push_back(val);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn len(&self) -> usize {
        self.size()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn front(&self) -> Option<&T> {
        self.data.front()
    }

    pub fn back(&self) -> Option<&T> {
        self.data.back()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    pub fn into_vec(self) -> Vec<T> {
        self.data.into_iter().collect()
    }
}

impl<T> From<Vec<T>> for Queue<T> {
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
}
