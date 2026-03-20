use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct ActionList {
    pub(crate) actions: Arc<Mutex<Vec<Arc<dyn Fn() + Send + Sync + 'static>>>>,
}

impl ActionList {
    pub fn do_action<F>(&self, f: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.actions.lock().unwrap().push(Arc::new(f));
    }
}
