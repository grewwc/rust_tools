use std::{
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    thread,
    time::{Duration, Instant},
};

use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};

use super::agent_routing::load_skill_manifests;
use crate::ai::{prompt::completion::CommandCompleter, skills};

/// 连续保存或解压技能包会产生多条事件；只在事件静默一小段时间后重新加载一次。
const SKILL_WATCH_DEBOUNCE: Duration = Duration::from_millis(300);
const SKILL_WATCH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const SKILL_WATCH_EVENT_QUEUE_SIZE: usize = 64;

/// 后台文件监听器。它直接更新补全缓存，并把完整快照发送给 driver 在安全点接管。
pub(super) struct SkillManifestWatcher {
    updates: Receiver<Arc<Vec<skills::SkillManifest>>>,
    shutdown: Arc<AtomicBool>,
}

impl SkillManifestWatcher {
    /// 取走累计的最新快照；旧快照已过期，无需让 driver 逐一应用。
    pub(super) fn take_latest(&mut self) -> Option<Arc<Vec<skills::SkillManifest>>> {
        let mut latest = None;
        while let Ok(update) = self.updates.try_recv() {
            latest = Some(update);
        }
        latest
    }
}

impl Drop for SkillManifestWatcher {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

/// 启动技能目录监听。`--no-skills` 模式不创建 watcher。
pub(super) fn start_skill_manifest_watcher(
    no_skills: bool,
) -> Result<Option<SkillManifestWatcher>, String> {
    if no_skills {
        return Ok(None);
    }

    let user_skills_dir = skills::skills_dir();
    let watch_roots = skills::skill_watch_roots();
    let (event_tx, event_rx) = mpsc::sync_channel(SKILL_WATCH_EVENT_QUEUE_SIZE);
    let mut watcher = notify::recommended_watcher(move |result: notify::Result<Event>| {
        if let Ok(event) = result {
            // 事件队列满时保留已有事件即可；随后仍会合并为一次 reload。
            let _ = event_tx.try_send(event);
        }
    })
    .map_err(|err| format!("创建文件监听器失败：{err}"))?;

    for root in &watch_roots {
        watcher
            .watch(root, RecursiveMode::Recursive)
            .map_err(|err| format!("监听技能目录 {} 失败：{err}", root.display()))?;
    }

    let (updates_tx, updates) = mpsc::channel();
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_shutdown = Arc::clone(&shutdown);
    let user_cache_dir = user_skills_dir.join(".cache");
    thread::Builder::new()
        .name("skill-manifest-watcher".to_string())
        .spawn(move || {
            run_skill_manifest_watcher(
                watcher,
                event_rx,
                updates_tx,
                worker_shutdown,
                watch_roots,
                user_cache_dir,
            );
        })
        .map_err(|err| format!("启动技能监听线程失败：{err}"))?;

    Ok(Some(SkillManifestWatcher { updates, shutdown }))
}

#[allow(clippy::too_many_arguments)]
fn run_skill_manifest_watcher(
    // watcher 必须在线程内持有，否则离开 start 函数时会停止接收文件系统事件。
    _watcher: RecommendedWatcher,
    event_rx: Receiver<Event>,
    updates_tx: mpsc::Sender<Arc<Vec<skills::SkillManifest>>>,
    shutdown: Arc<AtomicBool>,
    watch_roots: Vec<PathBuf>,
    user_cache_dir: PathBuf,
) {
    while !shutdown.load(Ordering::Acquire) {
        if !wait_for_skill_change(&event_rx, &shutdown, &watch_roots, &user_cache_dir) {
            continue;
        }
        debounce_skill_changes(&event_rx, &shutdown, &watch_roots, &user_cache_dir);
        if shutdown.load(Ordering::Acquire) {
            break;
        }

        let manifests = Arc::new(load_skill_manifests(false));
        CommandCompleter::set_skill_manifests(manifests.as_slice());
        if updates_tx.send(manifests).is_err() {
            break;
        }
    }
}

fn wait_for_skill_change(
    event_rx: &Receiver<Event>,
    shutdown: &AtomicBool,
    watch_roots: &[PathBuf],
    user_cache_dir: &Path,
) -> bool {
    loop {
        if shutdown.load(Ordering::Acquire) {
            return false;
        }
        match event_rx.recv_timeout(SKILL_WATCH_POLL_INTERVAL) {
            Ok(event) if is_relevant_skill_event(&event, watch_roots, user_cache_dir) => {
                return true;
            }
            Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return false,
        }
    }
}

fn debounce_skill_changes(
    event_rx: &Receiver<Event>,
    shutdown: &AtomicBool,
    watch_roots: &[PathBuf],
    user_cache_dir: &Path,
) {
    let mut last_relevant_event = Instant::now();
    loop {
        if shutdown.load(Ordering::Acquire) {
            return;
        }
        let elapsed = last_relevant_event.elapsed();
        if elapsed >= SKILL_WATCH_DEBOUNCE {
            return;
        }
        match event_rx.recv_timeout(SKILL_WATCH_DEBOUNCE - elapsed) {
            Ok(event) if is_relevant_skill_event(&event, watch_roots, user_cache_dir) => {
                last_relevant_event = Instant::now();
            }
            Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}

fn is_relevant_skill_event(event: &Event, watch_roots: &[PathBuf], user_cache_dir: &Path) -> bool {
    event.paths.iter().any(|path| {
        !path.starts_with(user_cache_dir) && watch_roots.iter().any(|root| path.starts_with(root))
    })
}

#[cfg(test)]
mod tests {
    use notify::{Event, EventKind};
    use std::path::PathBuf;

    use super::is_relevant_skill_event;

    #[test]
    fn ignores_skill_package_cache_events() {
        let skills_dir = PathBuf::from("/tmp/skills");
        let event = Event::new(EventKind::Any).add_path(skills_dir.join(".cache/package/SKILL.md"));

        assert!(!is_relevant_skill_event(
            &event,
            std::slice::from_ref(&skills_dir),
            &skills_dir.join(".cache"),
        ));
    }

    #[test]
    fn accepts_new_skill_file_event() {
        let skills_dir = PathBuf::from("/tmp/skills");
        let event = Event::new(EventKind::Any).add_path(skills_dir.join("new.skill"));

        assert!(is_relevant_skill_event(
            &event,
            std::slice::from_ref(&skills_dir),
            &skills_dir.join(".cache"),
        ));
    }
}
