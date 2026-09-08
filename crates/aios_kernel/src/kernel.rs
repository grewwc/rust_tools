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
    sync::{Arc, Mutex, OnceLock},
};

use crate::types::{FastMap, FastSet};

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
    Corrupted {
        expected_checksum: u64,
        actual_checksum: u64,
    },
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
            ShmReadError::Corrupted {
                expected_checksum,
                actual_checksum,
            } => {
                write!(
                    f,
                    "data corrupted (expected: {:#x}, actual: {:#x})",
                    expected_checksum, actual_checksum
                )
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
    pub parent_pid: Option<u64>, // Parent process PID
    pub name: String,
    pub goal: String, // Task goal description
    pub state: ProcessState,
    pub result: Option<String>, // Termination result
    pub mailbox: VecDeque<String>,
    pub max_mailbox_capacity: usize,
    pub pending_signals: VecDeque<Signal>, // Pending signals
    pub priority: u8,
    pub quota_turns: usize, // Max LLM turns allowed
    pub capabilities: ProcessCapabilities,
    pub is_foreground: bool,
    pub turns_used: usize, // Turns used
    pub created_at_tick: u64,
    pub process_group: Option<u64>, // Process group ID
    pub is_daemon: bool,            // Whether this is a daemon process
    pub max_restarts: usize,
    pub restart_count: usize,         // Restart count
    pub env: FastMap<String, String>, // Environment variables
    pub history_file: Option<PathBuf>,
    pub allowed_tools: FastSet<String>, // Allowed tools
    pub tool_calls_used: usize,         // Tool calls used
    pub working_dir: Option<PathBuf>,
    /// Structured resource limits (new). The legacy `quota_turns` field is kept
    /// as a view of max_turns; the two stay in sync when `limits` is modified
    /// via sys_rlimit_set.
    pub limits: crate::primitives::ResourceLimit,
    /// Structured resource usage accumulation (new). `turns_used` and
    /// `tool_calls_used` stay in sync with this.
    pub usage: crate::primitives::ResourceUsage,
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
    fn spawn(
        // Create a child process
        &mut self,
        parent_pid: Option<u64>,
        name: String,
        goal: String,
        priority: u8,
        quota_turns: usize,
        capabilities: Option<ProcessCapabilities>,
        allowed_tools: Option<FastSet<String>>,
    ) -> Result<u64, String>;
    fn wait_on(&mut self, target_pid: u64) -> Result<(), String>; // Wait for process termination
    fn wait_on_events(
        &mut self,
        event_ids: Vec<EventId>,
        policy: WaitPolicy,
        timeout_ticks: Option<u64>,
    ) -> Result<Option<u64>, String>; // Wait for external events
    fn send_ipc(&mut self, target_pid: u64, message: String) -> Result<(), String>; // Send IPC message
    fn read_mailbox(&mut self) -> Result<Vec<String>, String>; // Read mailbox
    fn set_env(&mut self, key: String, value: String) -> Result<(), String>; // Set environment variable
    fn get_env(&self, key: &str) -> Option<String>; // Get environment variable
    fn current_process_id(&self) -> Option<u64>; // Get current PID
    fn get_process(&self, pid: u64) -> Option<&Process>; // Get process info
    fn list_processes(&self) -> Vec<Process>; // List all processes
    fn sleep_current(&mut self, turns: u64) -> Result<u64, String>; // Sleep for N ticks
    fn kill_process(&mut self, target_pid: u64, reason: String) -> Result<(), String>; // Request process termination
    fn reap_process(&mut self, target_pid: u64) -> Result<String, String>; // Collect terminated process result
    fn signal_process(&mut self, target_pid: u64, signal: Signal) -> Result<(), String>; // Send signal
    fn set_process_group(&mut self, pid: u64, pgid: u64) -> Result<(), String>; // Set process group
    fn signal_process_group(&mut self, pgid: u64, signal: Signal) -> Result<usize, String>; // Signal a process group
    fn shm_create(&mut self, key: String, value: String) -> Result<(), String>; // Create shared memory
    fn shm_read(&self, key: &str) -> Result<String, ShmReadError>; // Read shared memory
    fn shm_read_degraded(&self, key: &str) -> Option<String>; // Fault-tolerant read
    fn shm_write(&mut self, key: String, value: String) -> Result<(), String>; // Write shared memory
    fn shm_delete(&mut self, key: &str) -> Result<(), String>; // Delete shared memory
    fn shm_health_check(&self) -> Vec<(String, ShmReadError)>; // Health check
    fn shm_cleanup_orphans(&mut self) -> usize; // Clean up orphaned shared memory
    fn set_working_dir(&mut self, dir: PathBuf) -> Result<(), String>; // Set working directory
    fn get_working_dir(&self) -> Option<PathBuf>; // Get working directory
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
    /// Pop the next ready process for scheduling (single).
    fn pop_ready(&mut self) -> Option<Process>;
    /// Pop multiple ready processes in batch (for concurrent execution).
    fn pop_all_ready(&mut self, max: usize) -> Vec<Process>;
    /// Set the PID of the currently executing process.
    fn set_current_pid(&mut self, pid: Option<u64>);
    /// Terminate the currently executing process (set state to Terminated with
    /// the given result string).
    fn terminate_current(&mut self, result: String);
    /// Get a mutable reference to the process with the given PID.
    fn get_process_mut(&mut self, pid: u64) -> Option<&mut Process>;
    /// Consume and clear the yield request flag, returning its previous value.
    /// Processes can request to yield the CPU via the yield_current tool.
    fn consume_yield_requested(&mut self) -> bool;
    /// Re-set the yield request flag.
    ///
    /// Some upper-layer wrappers (such as `epoll_wait_many`) first call
    /// `consume_yield_requested()` to read the pending state for their own
    /// decisions, which clears the kernel's yield intent. A later call to
    /// `consume_yield_requested()` in the turn loop would then read false and
    /// fail to hand control back to the scheduler (leaving the child agent
    /// stuck in Ready forever). After confirming that a suspension actually
    /// occurred, these wrappers must re-set the flag with this method so the
    /// yield intent is not lost.
    fn request_yield(&mut self);
    /// Check whether a kernel event has been marked as completed.
    fn event_is_completed(&self, event_id: EventId) -> bool;
    /// Drop a terminated process (non-waiting state).
    fn drop_terminated(&mut self, target_pid: u64) -> bool;
    /// Advance the scheduler tick, waking sleeping processes that are due.
    fn advance_tick(&mut self);
    /// Advance the scheduler tick in batch, waking all due processes.
    fn advance_ticks(&mut self, ticks: u64) {
        for _ in 0..ticks {
            self.advance_tick();
        }
    }
    /// The current scheduler tick.
    fn current_tick(&self) -> u64;
    /// The next tick that needs to be woken by a timer.
    fn next_wakeup_tick(&self) -> Option<u64>;
    /// Check whether there are ready processes (for scheduling decisions).
    fn has_ready(&self) -> bool;
    /// Return the number of processes in the ready queue.
    fn ready_count(&self) -> usize;
    /// Enable/disable round-robin scheduling (disabled by default).
    fn set_round_robin(&mut self, enabled: bool);
    /// Check whether round-robin scheduling is enabled.
    fn is_round_robin(&self) -> bool;
    /// Put the current process back into the ready queue (for round-robin
    /// scheduling).
    fn requeue_current(&mut self) -> bool;
    /// If the foreground process is Ready (woken up), take it out of the ready
    /// queue and set it to Running. Returns the activated foreground process,
    /// or None if there is none.
    fn pop_foreground_ready(&mut self) -> Option<Process>;
    /// Actively wake a process in Waiting/Sleeping state and write the wake
    /// reason into its mailbox. Returns true if the process was re-queued into
    /// the ready queue.
    fn wake_process(&mut self, pid: u64, message: String) -> bool;
    /// Process all pending signals of the current process, returning whether
    /// any signal was handled.
    fn process_pending_signals(&mut self) -> bool;
    /// Notify the kernel that some external events have reached a terminal
    /// state, to wake up processes waiting on them. Returns the list of PIDs
    /// that were woken.
    fn notify_events_completed(&mut self, completed_event_ids: &[EventId]) -> Vec<u64>;
    /// Increment the used turn count for the given process (for quota checks).
    fn increment_turns_used_for(&mut self, pid: u64);
    /// Increment the used tool call count for the given process (for quota
    /// checks).
    fn increment_tool_calls_used_for(&mut self, pid: u64);
    /// Check whether any daemon process needs a restart (stop restarting once
    /// max_restarts is exceeded). Returns the list of PIDs that need a restart.
    fn check_daemon_restart(&mut self) -> Vec<u64>;
    /// Clean up all resources of the given process (IPC, shared memory,
    /// environment variables, signals, etc.).
    fn cleanup_process_resources(&mut self, pid: u64);
}

/// Kernel trait - combines Syscall + KernelInternal + Futex + Trace.
/// Implement this trait to create a custom OS backend.
/// The LocalOS implementation provides the actual process table management.
pub trait Kernel:
    Syscall
    + KernelInternal
    + crate::primitives::FutexOps
    + crate::primitives::TraceOps
    + crate::primitives::EpollOps
    + crate::primitives::RlimitOps
    + crate::primitives::LlmOps
    + crate::primitives::VfsOps
    + crate::primitives::DaemonOps
    + crate::primitives::IpcOps
{
}

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

/// Runtime hook used to resolve the current async task's PID.
///
/// `aios_kernel` intentionally avoids committing to a specific async runtime.
/// Upper layers may register a provider backed by task-local state (for example,
/// `tokio::task_local!`) so `current_process_id()` keeps working across `.await`.
pub type CurrentPidProvider = fn() -> Option<u64>;

static CURRENT_PID_PROVIDER: OnceLock<CurrentPidProvider> = OnceLock::new();

pub fn register_current_pid_provider(provider: CurrentPidProvider) {
    let _ = CURRENT_PID_PROVIDER.set(provider);
}

pub fn current_task_pid() -> Option<u64> {
    CURRENT_PID_PROVIDER.get().and_then(|provider| provider())
}
