use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use rust_tools::commonw::{FastMap, FastSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    Ready,
    Running,
    Waiting { on_pid: u64 },
    Sleeping { until_tick: u64 },
    Stopped,
    Terminated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    SigTerm,
    SigStop,
    SigCont,
    SigKill,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShmReadError {
    NotFound,
    PermissionDenied { owner_pid: u64 },
    Corrupted { expected_checksum: u64, actual_checksum: u64 },
    OwnerTerminated { owner_pid: u64 },
}

impl std::fmt::Display for ShmReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShmReadError::NotFound => write!(f, "not found"),
            ShmReadError::PermissionDenied { owner_pid } => {
                write!(f, "permission denied (owner: {})", owner_pid)
            }
            ShmReadError::Corrupted { expected_checksum, actual_checksum } => {
                write!(f, "data corrupted (expected: {:#x}, actual: {:#x})", expected_checksum, actual_checksum)
            }
            ShmReadError::OwnerTerminated { owner_pid } => {
                write!(f, "owner process {} terminated", owner_pid)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessCapabilities {
    pub spawn: bool,
    pub wait: bool,
    pub ipc_send: bool,
    pub ipc_receive: bool,
    pub env_write: bool,
    pub manage_children: bool,
    pub sleep: bool,
    pub reap: bool,
    pub signal: bool,
}

impl ProcessCapabilities {
    pub fn full() -> Self {
        Self {
            spawn: true,
            wait: true,
            ipc_send: true,
            ipc_receive: true,
            env_write: true,
            manage_children: true,
            sleep: true,
            reap: true,
            signal: true,
        }
    }
}

pub const DEFAULT_MAILBOX_CAPACITY: usize = 64;

#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u64,
    pub parent_pid: Option<u64>,
    pub name: String,
    pub goal: String,
    pub state: ProcessState,
    pub result: Option<String>,
    pub mailbox: VecDeque<String>,
    pub max_mailbox_capacity: usize,
    pub pending_signals: VecDeque<Signal>,
    pub priority: u8,
    pub quota_turns: usize,
    pub capabilities: ProcessCapabilities,
    pub is_foreground: bool,
    pub turns_used: usize,
    pub created_at_tick: u64,
    pub process_group: Option<u64>,
    pub is_daemon: bool,
    pub max_restarts: usize,
    pub restart_count: usize,
    pub env: FastMap<String, String>,
    pub history_file: Option<PathBuf>,
    pub allowed_tools: FastSet<String>,
    pub tool_calls_used: usize,
    pub working_dir: Option<PathBuf>,
}

pub trait Syscall {
    fn spawn(
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        capabilities: Option<ProcessCapabilities>,
        allowed_tools: Option<FastSet<String>>,
    ) -> Result<u64, String>;
    fn wait_on(&mut self, target_pid: u64) -> Result<(), String>;
    fn send_ipc(&mut self, target_pid: u64, message: String) -> Result<(), String>;
    fn read_mailbox(&mut self) -> Result<Vec<String>, String>;
    fn set_env(&mut self, key: String, value: String) -> Result<(), String>;
    fn get_env(&self, key: &str) -> Option<String>;
    fn current_process_id(&self) -> Option<u64>;
    fn get_process(&self, pid: u64) -> Option<&Process>;
    fn list_processes(&self) -> Vec<Process>;
    fn sleep_current(&mut self, turns: u64) -> Result<u64, String>;
    fn kill_process(&mut self, target_pid: u64, reason: String) -> Result<(), String>;
    fn reap_process(&mut self, target_pid: u64) -> Result<String, String>;
    fn signal_process(&mut self, target_pid: u64, signal: Signal) -> Result<(), String>;
    fn set_process_group(&mut self, pid: u64, pgid: u64) -> Result<(), String>;
    fn signal_process_group(&mut self, pgid: u64, signal: Signal) -> Result<usize, String>;
    fn shm_create(&mut self, key: String, value: String) -> Result<(), String>;
    fn shm_read(&self, key: &str) -> Result<String, ShmReadError>;
    fn shm_read_degraded(&self, key: &str) -> Option<String>;
    fn shm_write(&mut self, key: String, value: String) -> Result<(), String>;
    fn shm_delete(&mut self, key: &str) -> Result<(), String>;
    fn shm_health_check(&self) -> Vec<(String, ShmReadError)>;
    fn shm_cleanup_orphans(&mut self) -> usize;
    fn set_working_dir(&mut self, dir: PathBuf) -> Result<(), String>;
    fn get_working_dir(&self) -> Option<PathBuf>;
    fn spawn_daemon(
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        max_restarts: usize,
    ) -> Result<u64, String>;
}

pub trait KernelInternal {
    fn begin_foreground(
        &mut self,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        allowed_tools: Option<FastSet<String>>,
    ) -> u64;
    fn pop_ready(&mut self) -> Option<Process>;
    fn pop_all_ready(&mut self, max: usize) -> Vec<Process>;
    fn set_current_pid(&mut self, pid: Option<u64>);
    fn terminate_current(&mut self, result: String);
    fn get_process_mut(&mut self, pid: u64) -> Option<&mut Process>;
    fn consume_yield_requested(&mut self) -> bool;
    fn drop_terminated(&mut self, target_pid: u64) -> bool;
    fn advance_tick(&mut self);
    fn has_ready(&self) -> bool;
    fn ready_count(&self) -> usize;
    fn set_round_robin(&mut self, enabled: bool);
    fn is_round_robin(&self) -> bool;
    fn requeue_current(&mut self) -> bool;
    fn process_pending_signals(&mut self) -> bool;
    fn increment_turns_used_for(&mut self, pid: u64);
    fn increment_tool_calls_used_for(&mut self, pid: u64);
    fn check_daemon_restart(&mut self) -> Vec<u64>;
    fn cleanup_process_resources(&mut self, pid: u64);
}

pub trait Kernel: Syscall + KernelInternal {}

pub type SharedKernel = Arc<Mutex<Box<dyn Kernel + Send>>>;

pub fn new_shared_kernel<K>(kernel: K) -> SharedKernel
where
    K: Kernel + Send + 'static,
{
    Arc::new(Mutex::new(Box::new(kernel)))
}

tokio::task_local! {
    pub static TASK_PID: Option<u64>;
}
