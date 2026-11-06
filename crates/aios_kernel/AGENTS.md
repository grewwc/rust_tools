# AGENTS.md - aios_kernel

## Scope

Standalone lib crate implementing the AIOS "process OS" for AI agents: process
table, scheduling, IPC mailboxes, shared memory, signals, plus primitive
facilities (see `src/primitives.rs`). Pure
`std` + `rustc-hash` - **no tokio / async runtime dependency**; the async side
(`tokio::task_local!` pid provider, waiters) lives in the `a` binary
(`src/bin/ai/driver/`).

## Layout

```text
src/kernel.rs       # Kernel / Syscall / KernelInternal traits + the shared process,
                    # signal, event, wait, capability, shm and pid-provider types
src/primitives.rs   # one ops trait per primitive facility (futex, trace, epoll, rlimit,
                    # llm-usage, vfs, daemon, ipc) + their types
src/local/          # LocalOS: single-machine implementation of every trait, split
                    # by facility (syscall, futex, ipc, vfs, daemon, ...)
src/types.rs        # FastMap / FastSet (rustc-hash re-exports)
```

## Build / Test

Crate targets: `cargo check -p aios_kernel`,
`cargo test -p aios_kernel <test_name>`.

## Invariants (do not break)

1. **Runtime-agnostic core.** `aios_kernel` must not depend on the root
   `rust_tools` crate, tokio, or any async runtime. Blocking waits are implemented
   agent-side; futex wait uses the waker-token pattern (or a sync `try_wait`), never
   a lock held across `.await`.
2. **One `Kernel` trait = Syscall + KernelInternal + all primitives ops.** New
   capability = a new trait in `primitives.rs` (or a method on an existing
   trait) + an impl on `LocalOS`; don't special-case callers.
3. **PID identity is runtime-provided.** `current_process_id()` resolves through
   `register_current_pid_provider` (task-local in `a`). Kernel code must not
   assume a global PID.
4. **Public surface.** Export kernel/primitive traits and shared types through
   `src/lib.rs`; keep implementation details private. Collections come from
   `crate::types::{FastMap, FastSet}` (`src/types.rs`), also exported by `src/lib.rs`.
5. **`LocalOS` state is guarded by the caller** via
   `SharedKernel = Arc<Mutex<Box<dyn Kernel + Send>>>` - kernel methods must
   not block or await.
6. **Process-table lifecycle.** Removing a process reattaches surviving children
   to an existing live foreground when available. A replacement foreground uses
   `KernelInternal::adopt_process_resources` to validate the exited owner and
   atomically adopt the child and its channels; the caller owns session scoping.
   Child-scoped ops (`kill_process` / `signal_process` / `reap_process` / `wait_on`)
   retain their owner checks. A target pid absent from the process table reports
   `"Process {pid} does not exist."` rather than a misleading "outside its scope"
   error.
