
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

impl<T> Stack<T> {
    pub fn push(&mut self, val: T) {
        self.data.push_back(val);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop_back()
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
}

