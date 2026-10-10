use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use super::{LocalOS, ShmReadError};
use crate::kernel::{
    EventId, KernelInternal, ProcessCapabilities, ProcessState, Signal, Syscall, WaitPolicy,
    WaitReason,
};

mod channel;
mod daemon;
mod epoll;
mod ipc;
mod primitives;
mod process;
mod scheduler;
mod shm;
mod vfs;
