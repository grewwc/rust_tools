pub struct Stack<T> {
    data: std::collections::LinkedList<T>,
}
impl<T> Stack<T> {
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
    pub fn push(&mut self, val: T) {
        self.data.push_back(val);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop_back()
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

    pub fn top(&self) -> Option<&T> {
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

impl<T> From<Vec<T>> for Stack<T> {
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
}
