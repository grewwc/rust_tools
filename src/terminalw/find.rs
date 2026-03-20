use std::{
    path::Path,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, AtomicI32, AtomicI64, Ordering},
    },
};

use crate::common::types::FastSet;

pub struct SyncSet {
    inner: Mutex<FastSet<String>>,
}

impl SyncSet {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FastSet::default()),
        }
    }

    pub fn add(&self, v: &str) {
        self.inner.lock().unwrap().insert(v.to_string());
    }

    pub fn contains(&self, v: &str) -> bool {
        self.inner.lock().unwrap().contains(v)
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }
}

impl Default for SyncSet {
    fn default() -> Self {
        Self::new()
    }
}

pub static FILE_NAMES_TO_CHECK: std::sync::LazyLock<SyncSet> =
    std::sync::LazyLock::new(SyncSet::new);
pub static FILE_NAMES_NOT_CHECK: std::sync::LazyLock<SyncSet> =
    std::sync::LazyLock::new(SyncSet::new);
pub static EXTENSIONS: std::sync::LazyLock<SyncSet> = std::sync::LazyLock::new(SyncSet::new);

pub static CHECK_EXTENSION: AtomicBool = AtomicBool::new(false);
pub static EXCLUDE: AtomicBool = AtomicBool::new(false);
pub static VERBOSE: AtomicBool = AtomicBool::new(false);
pub static NUM_PRINT: AtomicI64 = AtomicI64::new(5);
pub static COUNT: AtomicI64 = AtomicI64::new(0);
pub static MAX_LEVEL: AtomicI32 = AtomicI32::new(i32::MAX);

struct Semaphore {
    cap: usize,
    state: Mutex<usize>,
    cv: Condvar,
}

impl Semaphore {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            state: Mutex::new(0),
            cv: Condvar::new(),
        }
    }

    fn acquire(&self) {
        let mut used = self.state.lock().unwrap();
        while *used >= self.cap {
            used = self.cv.wait(used).unwrap();
        }
        *used += 1;
    }

    fn release(&self) {
        let mut used = self.state.lock().unwrap();
        *used = used.saturating_sub(1);
        self.cv.notify_one();
    }
}

static MAX_THREADS: std::sync::LazyLock<Mutex<Arc<Semaphore>>> =
    std::sync::LazyLock::new(|| Mutex::new(Arc::new(Semaphore::new(4))));

pub fn change_threads(num: usize) {
    let mut lock = MAX_THREADS.lock().unwrap();
    *lock = Arc::new(Semaphore::new(num.max(1)));
}

pub struct WaitGroup {
    inner: Arc<(Mutex<usize>, Condvar)>,
}

impl WaitGroup {
    pub fn new() -> Self {
        Self {
            inner: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }

    pub fn add(&self, n: usize) {
        let (m, _) = &*self.inner;
        let mut v = m.lock().unwrap();
        *v += n;
    }

    pub fn done(&self) {
        let (m, cv) = &*self.inner;
        let mut v = m.lock().unwrap();
        *v = v.saturating_sub(1);
        if *v == 0 {
            cv.notify_all();
        }
    }

    pub fn wait(&self) {
        let (m, cv) = &*self.inner;
        let mut v = m.lock().unwrap();
        while *v != 0 {
            v = cv.wait(v).unwrap();
        }
    }
}

impl Default for WaitGroup {
    fn default() -> Self {
        Self::new()
    }
}

fn is_probably_text_file(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    if bytes.is_empty() {
        return true;
    }
    let sample = &bytes[..bytes.len().min(8192)];
    if sample.contains(&0) {
        return false;
    }
    std::str::from_utf8(sample).is_ok()
}

pub fn find<F>(root_dir: &str, task: Arc<F>, wg: Arc<WaitGroup>, level: i32)
where
    F: Fn(String) + Send + Sync + 'static,
{
    wg.add(1);
    let root = root_dir.to_string();
    std::thread::spawn(move || {
        find_impl(&root, &task, &wg, level);
        wg.done();
    });
}

fn find_impl<F>(root_dir: &str, task: &Arc<F>, wg: &Arc<WaitGroup>, level: i32)
where
    F: Fn(String) + Send + Sync + 'static,
{
    if level > MAX_LEVEL.load(Ordering::Relaxed) {
        return;
    }
    let sem = { Arc::clone(&*MAX_THREADS.lock().unwrap()) };
    sem.acquire();
    let _guard = ScopeGuard::new(move || sem.release());

    if COUNT.load(Ordering::Relaxed) >= NUM_PRINT.load(Ordering::Relaxed) {
        return;
    }

    let Ok(entries) = std::fs::read_dir(root_dir) else {
        return;
    };

    for entry in entries.flatten() {
        if COUNT.load(Ordering::Relaxed) >= NUM_PRINT.load(Ordering::Relaxed) {
            return;
        }

        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if (!FILE_NAMES_TO_CHECK.is_empty() && !FILE_NAMES_TO_CHECK.contains(name))
            || FILE_NAMES_NOT_CHECK.contains(name)
        {
            continue;
        }

        if path.is_dir() {
            let next = path.to_string_lossy().to_string();
            let task = Arc::clone(task);
            let wg2 = Arc::clone(wg);
            find(&next, task, wg2, level + 1);
            continue;
        }

        if !is_probably_text_file(&path) {
            continue;
        }

        if !CHECK_EXTENSION.load(Ordering::Relaxed) {
            task(path.to_string_lossy().to_string());
        } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            let e = format!(".{ext}");
            if EXTENSIONS.contains(&e) {
                task(path.to_string_lossy().to_string());
            }
        }
    }
}

struct ScopeGuard<F: FnOnce()> {
    f: Option<F>,
}

impl<F: FnOnce()> ScopeGuard<F> {
    fn new(f: F) -> Self {
        Self { f: Some(f) }
    }
}

impl<F: FnOnce()> Drop for ScopeGuard<F> {
    fn drop(&mut self) {
        if let Some(f) = self.f.take() {
            f();
        }
    }
}
