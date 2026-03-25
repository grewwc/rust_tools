use std::collections::VecDeque;

#[derive(Clone)]
pub struct DequeList<T> {
    data: VecDeque<T>,
}

impl<T> DequeList<T> {
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(cap),
        }
    }

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    pub fn push_back(&mut self, v: T) {
        self.data.push_back(v);
    }

    pub fn push_front(&mut self, v: T) {
        self.data.push_front(v);
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.data.pop_front()
    }

    pub fn pop_back(&mut self) -> Option<T> {
        self.data.pop_back()
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

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = T>,
    {
        self.data.extend(iter);
    }

    pub fn into_vec_deque(self) -> VecDeque<T> {
        self.data
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().cloned().collect()
    }

    pub fn remove_first<F>(&mut self, pred: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        if let Some(pos) = self.data.iter().position(pred) {
            self.data.remove(pos);
            return true;
        }
        false
    }
}

impl<T> Default for DequeList<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> From<VecDeque<T>> for DequeList<T> {
    fn from(data: VecDeque<T>) -> Self {
        Self { data }
    }
}

impl<T> From<Vec<T>> for DequeList<T> {
    fn from(data: Vec<T>) -> Self {
        Self {
            data: VecDeque::from(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DequeList;

    #[test]
    fn test_basic_push_pop_and_iter() {
        let mut d = DequeList::new();
        d.push_back(2);
        d.push_front(1);
        d.push_back(3);
        assert_eq!(d.front(), Some(&1));
        assert_eq!(d.back(), Some(&3));
        assert_eq!(d.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(d.pop_front(), Some(1));
        assert_eq!(d.pop_back(), Some(3));
        assert_eq!(d.to_vec(), vec![2]);
    }
}
