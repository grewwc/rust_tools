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

    pub fn clear(&mut self) {
        self.data.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn push_back(&mut self, v: T) {
        self.data.push_back(v);
    }

    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.data.iter().cloned().collect()
    }

    pub fn remove_first<F>(&mut self, mut pred: F) -> bool
    where
        F: FnMut(&T) -> bool,
    {
        if let Some(pos) = self.data.iter().position(|x| pred(x)) {
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
