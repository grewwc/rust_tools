use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};

use crate::ai::tools::os_tools::GLOBAL_OS;
use crate::ai::tools::storage::file_store::current_session_assets_dir;
use crate::ai::{
    agents::{self, AgentManifest, AgentModelTier},
    models,
    tools::common::{
        ToolHistoryPolicy, ToolHistoryPolicyRegistration, ToolLossyCompressPolicy, ToolPrunePolicy,
    },
    tools::common::{ToolRegistration, ToolSpec},
    tools::registry::common::current_process_tool_cancel_futex,
};
use aios_kernel::SharedKernel;
use aios_kernel::{
    kernel::{EventId, Kernel, ProcessState, WaitPolicy},
    primitives::{
        ChannelId, ChannelOwnerTag, EpollEventMask, EpollSource, EpollWaitResult, FutexAddr,
        IpcRecvResult,
    },
};
use rust_tools::cw::{SkipMap, SkipSet};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

mod agent_team;

mod collect;
mod dispatch;
mod epoll;
mod outstanding;
mod progress;
mod registry;
mod spawn;
mod types;
mod wait;

// `collect` items are `pub(super)` (task_tools-tree visible), so the re-export
// visibility must match exactly: broader would not re-export (warning), and it
// would leak `AgentManifest` (private to the tree) into a public interface.
pub(in crate::ai::tools::task_tools) use collect::*;
pub(crate) use dispatch::*;
pub(crate) use epoll::*;
pub(crate) use outstanding::*;
pub(crate) use progress::*;
pub(crate) use registry::*;
pub(crate) use spawn::*;
pub(crate) use types::*;
pub(crate) use wait::*;

#[cfg(test)]
#[path = "../task_tools_tests.rs"]
mod tests;
