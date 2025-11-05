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

impl<T> Queue<T> {
    pub fn enqueue(&mut self, val: T) {
        self.data.push_back(val);
    }

    pub fn dequeue(&mut self) -> Option<T> {
        self.data.pop_front()
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
}

