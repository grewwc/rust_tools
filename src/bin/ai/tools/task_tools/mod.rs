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

mod backend;
pub(crate) mod backend_session;
pub(crate) mod backends;
mod collect;
mod dispatch;
mod epoll;
mod outstanding;
mod progress;
mod quiet;
mod recovery;
mod registry;
mod spawn;
mod traecli;
mod traecli_models;
mod types;
mod wait;

// `collect` items are `pub(super)` (task_tools-tree visible), so the re-export
// visibility must match exactly: broader would not re-export (warning), and it
// would leak `AgentManifest` (private to the tree) into a public interface.
pub(crate) use backend::*;
pub(crate) use collect::publish_task_result;
pub(in crate::ai::tools::task_tools) use collect::*;
pub(crate) use dispatch::*;
pub(crate) use epoll::*;
pub(crate) use outstanding::*;
pub(crate) use progress::*;
pub(crate) use quiet::{is_agent_team_mode, is_agent_team_wait_deadline_message};
pub(crate) use recovery::*;
pub(crate) use registry::*;
pub(crate) use spawn::*;
#[cfg(test)]
pub(crate) use traecli::{
    prepare_traecli_continuation, record_traecli_session, resolve_continue_target,
};
pub(crate) use traecli_models::backend_model_names;
#[cfg(test)]
pub(crate) use traecli_models::with_traecli_model_names_for_test;
pub(crate) use types::*;
pub(crate) use wait::*;

#[cfg(test)]
#[path = "../task_tools_tests.rs"]
mod tests;
