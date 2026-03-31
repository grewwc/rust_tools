use std::sync::{Arc, Mutex};

pub(crate) type ActionFn = Arc<dyn Fn() + Send + Sync + 'static>;
pub(crate) type ActionFnList = Arc<Mutex<Vec<ActionFn>>>;

#[derive(Clone)]
pub struct ActionList {
    pub(crate) actions: ActionFnList,
}

impl ActionList {
    pub fn do_action<F>(&self, f: F)
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.actions.lock().unwrap().push(Arc::new(f));
    }
}
