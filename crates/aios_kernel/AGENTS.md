# AGENTS.md - aios_kernel

## Scope

Standalone lib crate implementing the AIOS "process OS" for AI agents: process
table, scheduling, IPC mailboxes, shared memory, signals, plus the primitive
facilities (futex, trace, epoll, rlimit, llm-usage, vfs, daemon, ipc). Pure
`std` + `rustc-hash` - **no tokio / async runtime dependency**; the async side
(`tokio::task_local!` pid provider, waiters) lives in the `a` binary
(`src/bin/ai/driver/`).

## Layout

```text
src/kernel.rs       # Kernel / Syscall / KernelInternal traits, Process, ProcessState,
                    # Signal, EventId, WaitPolicy/WaitReason, ShmReadError,
                    # ProcessCapabilities, SharedKernel, CurrentPidProvider
src/primitives.rs   # FutexOps, TraceOps, EpollOps, RlimitOps, LlmOps, VfsOps,
                    # DaemonOps, IpcOps + their types (ResourceLimit/Usage, ...)
src/local.rs        # LocalOS: single-machine implementation of every trait (~6.4K lines)
src/types.rs        # FastMap / FastSet (rustc-hash re-exports)
```

## Build / Test

```bash
cargo check -p aios_kernel
cargo test -p aios_kernel test_name
```

## Invariants (do not break)

1. **Runtime-agnostic core.** `aios_kernel` must not depend on tokio or any
   async runtime. Blocking waits are implemented agent-side; futex wait uses the
   waker-token pattern (or a sync `try_wait`), never a lock held across `.await`.
2. **One `Kernel` trait = Syscall + KernelInternal + all primitives ops.** New
   capability = a new trait in `primitives.rs` (or a method on an existing
   trait) + an impl on `LocalOS`; don't special-case callers.
3. **PID identity is runtime-provided.** `current_process_id()` resolves through
   `register_current_pid_provider` (task-local in `a`). Kernel code must not
   assume a global PID.
4. **Collections**: use `FastMap`/`FastSet` from `types` (rustc-hash).
5. **Module boundaries**: `pub(super)` / `pub(crate)`; keep the public surface
   minimal (`lib.rs` re-exports the four modules).
6. **`LocalOS` state is guarded by the caller** via
   `SharedKernel = Arc<Mutex<Box<dyn Kernel + Send>>>` - kernel methods must
   not block or await.