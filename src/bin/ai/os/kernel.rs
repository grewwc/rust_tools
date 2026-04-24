// =============================================================================
// AIOS Kernel - Agent Operating System Core
// =============================================================================
// This module implements a process-based OS for AI agents, providing:
// - Process management (spawn, wait, kill, reap)
// - Inter-process communication (IPC via mailboxes)
// - Signal handling (SIGTERM, SIGSTOP, SIGCONT, SIGKILL)
// - Shared memory (shm_create/read/write/delete)
// - Process scheduling (ready/running/waiting/sleeping states)
// =============================================================================

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use rust_tools::commonw::{FastMap, FastSet};

/// Process lifecycle states - similar to OS process states
/// Controls process scheduling and execution flow
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitPolicy {
    Any,
    All,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EventId(u64);

impl EventId {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl std::fmt::Display for EventId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "evt_{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaitReason {
    /// Wait until another process terminates.
    ProcessExit { on_pid: u64 },
    /// Wait for one or more external events to reach a terminal state.
    /// Runtime layers are responsible for mapping domain-specific async work
    /// (tool tasks, background jobs, etc.) onto these opaque event ids.
    Events {
        event_ids: Vec<EventId>,
        policy: WaitPolicy,
        timeout_tick: Option<u64>,
    },
}

/// Process lifecycle states - similar to OS process states
/// Controls process scheduling and execution flow
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcessState {
    /// Process is ready to be scheduled for execution
    Ready,
    /// Process is currently executing (in an LLM turn)
    Running,
    /// Process is blocked waiting for an external condition to be satisfied.
    /// Examples:
    /// - another process terminates
    /// - one or more external events finish
    Waiting { reason: WaitReason },
    /// Process is sleeping for a number of scheduler ticks
    /// Used by sleep_current syscall - pause execution for N ticks
    Sleeping { until_tick: u64 },
    /// Process is stopped - typically receives SIGSTOP signal
    /// Can be resumed with SIGCONT
    Stopped,
    /// Process has terminated - awaiting parent to call reap()
    /// result field contains termination reason
    Terminated,
}

/// Signals - process control signals, mirroring POSIX signals
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    /// Cooperative cancellation request for the current turn/tool execution.
    SigCancel,
    /// Graceful termination request - process should clean up and exit
    SigTerm,
    /// Stop execution - process pauses (similar to Ctrl+Z)
    SigStop,
    /// Resume execution - continue from where it stopped
    SigCont,
    /// Immediate termination - cannot be caught or ignored
    SigKill,
}

/// Errors when reading from shared memory region
#[derive(Debug, Clone, PartialEq)]
pub enum ShmReadError {
    /// The requested key does not exist
    NotFound,
    /// Caller is not the owner of this shared memory region
    PermissionDenied { owner_pid: u64 },
    /// Data corruption detected via checksum mismatch
    Corrupted { expected_checksum: u64, actual_checksum: u64 },
    /// The process that owns this region has terminated
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

/// Process capabilities - capability-based security for processes
/// Similar to Linux capabilities, enables fine-grained permission control.
/// Each process can only perform syscalls matching its enabled capabilities.
/// This prevents malicious or buggy agents from performing unauthorized actions.
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

/// Default mailbox capacity for new processes
pub const DEFAULT_MAILBOX_CAPACITY: usize = 64;

/// AIOS Process - a LLM-driven execution unit.
/// Unlike traditional OS processes that execute CPU instructions,
/// AIOS processes execute LLM turns until goal completion or quota exhaustion.
/// Key concepts:
///   - goal: The task this process should accomplish (set at spawn time)
///   - mailbox: IPC message queue - receives messages from other processes
///   - quota_turns: Max LLM turns allowed - resource limiting per process
///   - capabilities: What syscalls this process can invoke (security)
///   - is_foreground: Whether this is the interactive foreground process
#[derive(Debug, Clone)]
pub struct Process {
    pub pid: u64,
    pub parent_pid: Option<u64>,  // 父进程 PID
    pub name: String,
    pub goal: String,  // 任务描述
    pub state: ProcessState,
    pub result: Option<String>,  // 终止结果
    pub mailbox: VecDeque<String>,
    pub max_mailbox_capacity: usize,
    pub pending_signals: VecDeque<Signal>,  // 待处理信号
    pub priority: u8,
    pub quota_turns: usize,  // 最大 LLM turn 数
    pub capabilities: ProcessCapabilities,
    pub is_foreground: bool,
    pub turns_used: usize,  // 已使用 turn 数
    pub created_at_tick: u64,
    pub process_group: Option<u64>,  // 进程组 ID
    pub is_daemon: bool,  // 是否守护进程
    pub max_restarts: usize,
    pub restart_count: usize,  // 重启计数
    pub env: FastMap<String, String>,  // 环境变量
    pub history_file: Option<PathBuf>,
    pub allowed_tools: FastSet<String>,  // 允许的工具
    pub tool_calls_used: usize,  // 已使用 tool call 数
    pub working_dir: Option<PathBuf>,
}

/// Syscall trait - system calls available to AIOS processes.
/// These are invoked by AI agents via tool calls to interact with the OS.
/// 
/// Process management:
///   - spawn(): Create a new child process (subagent)
///   - wait_on(): Block until another process terminates
///   - wait_on_events(): Block until external events reach the desired completion condition
///   - kill_process(): Request termination of another process
///   - reap_process(): Collect terminated process result
/// 
/// IPC:
///   - send_ipc(): Send message to another process's mailbox
///   - read_mailbox(): Read messages from own mailbox
/// 
/// Shared memory:
///   - shm_create(): Create new shared memory region (key/value)
///   - shm_read(): Read from shared memory (owner only)
///   - shm_write(): Write to shared memory (owner only)
///   - shm_delete(): Delete shared memory region
/// 
/// Environment:
///   - set_env(): Set environment variable
///   - set_working_dir(): Change current working directory
pub trait Syscall {
    fn spawn(  // 创建子进程
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        capabilities: Option<ProcessCapabilities>,
        allowed_tools: Option<FastSet<String>>,
    ) -> Result<u64, String>;
    fn wait_on(&mut self, target_pid: u64) -> Result<(), String>;  // 等待进程终止
    fn wait_on_events(
        &mut self,
        event_ids: Vec<EventId>,
        policy: WaitPolicy,
        timeout_ticks: Option<u64>,
    ) -> Result<Option<u64>, String>;  // 等待外部事件
    fn send_ipc(&mut self, target_pid: u64, message: String) -> Result<(), String>;  // 发送 IPC 消息
    fn read_mailbox(&mut self) -> Result<Vec<String>, String>;  // 读取邮箱
    fn set_env(&mut self, key: String, value: String) -> Result<(), String>;  // 设置环境变量
    fn get_env(&self, key: &str) -> Option<String>;  // 获取环境变量
    fn current_process_id(&self) -> Option<u64>;  // 获取当前 PID
    fn get_process(&self, pid: u64) -> Option<&Process>;  // 获取进程信息
    fn list_processes(&self) -> Vec<Process>;  // 列出所有进程
    fn sleep_current(&mut self, turns: u64) -> Result<u64, String>;  // 睡眠 N 个 tick
    fn kill_process(&mut self, target_pid: u64, reason: String) -> Result<(), String>;  // 请求终止进程
    fn reap_process(&mut self, target_pid: u64) -> Result<String, String>;  // 收集终止进程结果
    fn signal_process(&mut self, target_pid: u64, signal: Signal) -> Result<(), String>;  // 发送信号
    fn set_process_group(&mut self, pid: u64, pgid: u64) -> Result<(), String>;  // 设置进程组
    fn signal_process_group(&mut self, pgid: u64, signal: Signal) -> Result<usize, String>;  // 组播信号
    fn shm_create(&mut self, key: String, value: String) -> Result<(), String>;  // 创建共享内存
    fn shm_read(&self, key: &str) -> Result<String, ShmReadError>;  // 读取共享内存
    fn shm_read_degraded(&self, key: &str) -> Option<String>;  // 容错读取
    fn shm_write(&mut self, key: String, value: String) -> Result<(), String>;  // 写入共享��存
    fn shm_delete(&mut self, key: &str) -> Result<(), String>;  // 删除共享内存
    fn shm_health_check(&self) -> Vec<(String, ShmReadError)>;  // 健康检查
    fn shm_cleanup_orphans(&mut self) -> usize;  // 清理孤立共享内存
    fn set_working_dir(&mut self, dir: PathBuf) -> Result<(), String>;  // 设置工作目录
    fn get_working_dir(&self) -> Option<PathBuf>;  // 获取工作目录
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

/// KernelInternal - internal kernel operations not exposed as syscalls.
/// Used for process scheduling, state transitions, and cleanup.
/// These are called by the turn runtime and driver, not by AI agents directly.
/// 
/// Scheduling:
///   - begin_foreground(): Create foreground process for interactive input
///   - pop_ready(): Get next ready process (scheduling)
///   - pop_all_ready(): Get all ready processes (batch scheduling)
///   - requeue_current(): Put current process back in ready queue (round-robin)
pub trait KernelInternal {
    fn begin_foreground(
        &mut self,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        allowed_tools: Option<FastSet<String>>,
    ) -> u64;
    /// 弹出下一个就绪进程用于调度（单个）
    fn pop_ready(&mut self) -> Option<Process>;
    /// 批量弹出多个就绪进程（用于并发执行）
    fn pop_all_ready(&mut self, max: usize) -> Vec<Process>;
    /// 设置当前正在执行的进程 PID
    fn set_current_pid(&mut self, pid: Option<u64>);
    /// 终止当前正在执行的进程（设置状态为 Terminated，结果为传入的字符串）
    fn terminate_current(&mut self, result: String);
    /// 获取指定 PID 的进程可变引用
    fn get_process_mut(&mut self, pid: u64) -> Option<&mut Process>;
    /// 消费并清除 yield 请求标志，返回之前的值
    /// 进程可通过 yield_current 工具请求让出 CPU
    fn consume_yield_requested(&mut self) -> bool;
    /// 删除已终止的进程（非等待状态）
    fn drop_terminated(&mut self, target_pid: u64) -> bool;
    /// 推进调度器 tick，唤醒到期睡眠进程
    fn advance_tick(&mut self);
    /// 检查是否有就绪进程（用于调度决策）
    fn has_ready(&self) -> bool;
    /// 返回就绪队列中的进程数量
    fn ready_count(&self) -> usize;
    /// 启用/禁用轮转调度（默认禁用）
    fn set_round_robin(&mut self, enabled: bool);
    /// 检查是否启用了轮转调度
    fn is_round_robin(&self) -> bool;
    /// 将当前进程重新放回就绪队列（用于轮转调度）
    fn requeue_current(&mut self) -> bool;
    /// 处理当前进程的所有待处理信号，返回是否处理了信号
    fn process_pending_signals(&mut self) -> bool;
    /// 通知 kernel 某些外部事件已进入终态，用于唤醒等待这些事件的进程。
    /// 返回被唤醒的 PID 列表。
    fn notify_events_completed(&mut self, completed_event_ids: &[EventId]) -> Vec<u64>;
    /// 为指定进程增加已使用 turn 数（用于配额检查）
    fn increment_turns_used_for(&mut self, pid: u64);
    /// 为指定进程增加已使用 tool call 数（用于配额检查）
    fn increment_tool_calls_used_for(&mut self, pid: u64);
    /// 检查所有守护进程是否需要重启（当超过 max_restarts 时停止重启）
    /// 返回需要重启的进程 PID 列表
    fn check_daemon_restart(&mut self) -> Vec<u64>;
    /// 清理指定进程的所有资源（IPC、共享内存、环境变量、信号等）
    fn cleanup_process_resources(&mut self, pid: u64);
}

/// Kernel trait - combines Syscall + KernelInternal.
/// Implement this trait to create a custom OS backend.
/// The LocalOS implementation provides the actual process table management.
pub trait Kernel: Syscall + KernelInternal {}

/// SharedKernel - shared reference to kernel implementation.
/// Wrapped in Arc<Mutex<>> for thread-safe access from multiple async tasks.
pub type SharedKernel = Arc<Mutex<Box<dyn Kernel + Send>>>;

/// Create a new shared kernel from any Kernel implementation.
pub fn new_shared_kernel<K>(kernel: K) -> SharedKernel
where
    K: Kernel + Send + 'static,
{
    Arc::new(Mutex::new(Box::new(kernel)))
}

// Task-local storage for the current process ID.
// Used by spawn_process() to scope async blocks to specific processes.
// This allows the turn runtime to know which process is currently executing.
tokio::task_local! {
    pub static TASK_PID: Option<u64>;
}
